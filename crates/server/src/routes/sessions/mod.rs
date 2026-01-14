pub mod queue;
pub mod review;

use std::str::FromStr;

use axum::{
    Extension, Json, Router,
    extract::{Query, State},
    middleware::from_fn_with_state,
    response::Json as ResponseJson,
    routing::{get, post},
};
use db::models::{
    execution_process::{ExecutionProcess, ExecutionProcessRunReason, ExecutionProcessStatus},
    execution_process_logs::ExecutionProcessLogs,
    scratch::{Scratch, ScratchType},
    session::{CreateSession, Session},
    workspace::{Workspace, WorkspaceError},
    workspace_repo::WorkspaceRepo,
};
use deployment::Deployment;
use executors::{
    actions::{
        ExecutorAction, ExecutorActionType, coding_agent_follow_up::CodingAgentFollowUpRequest,
    },
    executors::BaseCodingAgent,
    logs::utils::patch::PatchType,
    profile::ExecutorProfileId,
};
use json_patch::patch as apply_patch;
use serde::Serialize;
use serde_json::json;
use serde::Deserialize;
use services::services::container::ContainerService;
use ts_rs::TS;
use utils::response::ApiResponse;
use uuid::Uuid;

use crate::{
    DeploymentImpl, error::ApiError, middleware::load_session_middleware,
    routes::task_attempts::util::restore_worktrees_to_process,
};

#[derive(Debug, Deserialize)]
pub struct SessionQuery {
    pub workspace_id: Uuid,
}

#[derive(Debug, Deserialize, TS)]
pub struct CreateSessionRequest {
    pub workspace_id: Uuid,
    pub executor: Option<String>,
}

pub async fn get_sessions(
    State(deployment): State<DeploymentImpl>,
    Query(query): Query<SessionQuery>,
) -> Result<ResponseJson<ApiResponse<Vec<Session>>>, ApiError> {
    let pool = &deployment.db().pool;
    let sessions = Session::find_by_workspace_id(pool, query.workspace_id).await?;
    Ok(ResponseJson(ApiResponse::success(sessions)))
}

pub async fn get_session(
    Extension(session): Extension<Session>,
) -> Result<ResponseJson<ApiResponse<Session>>, ApiError> {
    Ok(ResponseJson(ApiResponse::success(session)))
}

#[derive(Serialize)]
pub struct ConversationProcessEntries {
    pub execution_process: ExecutionProcess,
    pub entries: Vec<PatchType>,
}

#[derive(Serialize)]
pub struct SessionConversationResponse {
    pub processes: Vec<ConversationProcessEntries>,
}

pub async fn get_session_conversation(
    Extension(session): Extension<Session>,
    State(deployment): State<DeploymentImpl>,
) -> Result<ResponseJson<ApiResponse<SessionConversationResponse>>, ApiError> {
    let pool = &deployment.db().pool;
    let execution_processes =
        ExecutionProcess::find_by_session_id(pool, session.id, false).await?;

    let mut processes = Vec::new();
    for execution_process in execution_processes {
        if !matches!(
            execution_process.run_reason,
            ExecutionProcessRunReason::SetupScript
                | ExecutionProcessRunReason::CleanupScript
                | ExecutionProcessRunReason::CodingAgent
        ) {
            continue;
        }

        let entries = load_entries_for_process(&deployment, &execution_process).await?;

        processes.push(ConversationProcessEntries {
            execution_process,
            entries,
        });
    }

    Ok(ResponseJson(ApiResponse::success(SessionConversationResponse {
        processes,
    })))
}

pub async fn create_session(
    State(deployment): State<DeploymentImpl>,
    Json(payload): Json<CreateSessionRequest>,
) -> Result<ResponseJson<ApiResponse<Session>>, ApiError> {
    let pool = &deployment.db().pool;

    // Verify workspace exists
    let _workspace = Workspace::find_by_id(pool, payload.workspace_id)
        .await?
        .ok_or(ApiError::Workspace(WorkspaceError::ValidationError(
            "Workspace not found".to_string(),
        )))?;

    let session = Session::create(
        pool,
        &CreateSession {
            executor: payload.executor,
        },
        Uuid::new_v4(),
        payload.workspace_id,
    )
    .await?;

    Ok(ResponseJson(ApiResponse::success(session)))
}

#[derive(Debug, Deserialize, TS)]
pub struct CreateFollowUpAttempt {
    pub prompt: String,
    pub variant: Option<String>,
    pub retry_process_id: Option<Uuid>,
    pub force_when_dirty: Option<bool>,
    pub perform_git_reset: Option<bool>,
}

pub async fn follow_up(
    Extension(session): Extension<Session>,
    State(deployment): State<DeploymentImpl>,
    Json(payload): Json<CreateFollowUpAttempt>,
) -> Result<ResponseJson<ApiResponse<ExecutionProcess>>, ApiError> {
    let pool = &deployment.db().pool;

    // Load workspace from session
    let workspace = Workspace::find_by_id(pool, session.workspace_id)
        .await?
        .ok_or(ApiError::Workspace(WorkspaceError::ValidationError(
            "Workspace not found".to_string(),
        )))?;

    tracing::info!("{:?}", workspace);

    deployment
        .container()
        .ensure_container_exists(&workspace)
        .await?;

    // Get executor from the latest CodingAgent process, or fall back to session's executor
    let base_executor =
        match ExecutionProcess::latest_executor_profile_for_session(pool, session.id).await? {
            Some(profile) => profile.executor,
            None => {
                // No prior execution - use session's executor field
                let executor_str = session.executor.as_ref().ok_or_else(|| {
                    ApiError::Workspace(WorkspaceError::ValidationError(
                        "No prior execution and no executor configured on session".to_string(),
                    ))
                })?;
                BaseCodingAgent::from_str(&executor_str.replace('-', "_").to_ascii_uppercase())
                    .map_err(|_| {
                        ApiError::Workspace(WorkspaceError::ValidationError(format!(
                            "Invalid executor: {}",
                            executor_str
                        )))
                    })?
            }
        };

    let executor_profile_id = ExecutorProfileId {
        executor: base_executor,
        variant: payload.variant,
    };

    // If retry settings provided, perform replace-logic before proceeding
    if let Some(proc_id) = payload.retry_process_id {
        // Validate process belongs to this session
        let process =
            ExecutionProcess::find_by_id(pool, proc_id)
                .await?
                .ok_or(ApiError::Workspace(WorkspaceError::ValidationError(
                    "Process not found".to_string(),
                )))?;
        if process.session_id != session.id {
            return Err(ApiError::Workspace(WorkspaceError::ValidationError(
                "Process does not belong to this session".to_string(),
            )));
        }

        // Reset all repository worktrees to the state before the target process
        let force_when_dirty = payload.force_when_dirty.unwrap_or(false);
        let perform_git_reset = payload.perform_git_reset.unwrap_or(true);
        restore_worktrees_to_process(
            &deployment,
            pool,
            &workspace,
            proc_id,
            perform_git_reset,
            force_when_dirty,
        )
        .await?;

        // Stop any running processes for this workspace (except dev server)
        deployment.container().try_stop(&workspace, false).await;

        // Soft-drop the target process and all later processes in that session
        let _ = ExecutionProcess::drop_at_and_after(pool, process.session_id, proc_id).await?;
    }

    let latest_agent_session_id =
        ExecutionProcess::find_latest_coding_agent_turn_session_id(pool, session.id).await?;

    let prompt = payload.prompt;

    let repos = WorkspaceRepo::find_repos_for_workspace(pool, workspace.id).await?;
    let cleanup_action = deployment.container().cleanup_actions_for_repos(&repos);

    let working_dir = workspace
        .agent_working_dir
        .as_ref()
        .filter(|dir| !dir.is_empty())
        .cloned();

    let action_type = if let Some(agent_session_id) = latest_agent_session_id {
        ExecutorActionType::CodingAgentFollowUpRequest(CodingAgentFollowUpRequest {
            prompt: prompt.clone(),
            session_id: agent_session_id,
            executor_profile_id: executor_profile_id.clone(),
            working_dir: working_dir.clone(),
        })
    } else {
        ExecutorActionType::CodingAgentInitialRequest(
            executors::actions::coding_agent_initial::CodingAgentInitialRequest {
                prompt,
                executor_profile_id: executor_profile_id.clone(),
                working_dir,
            },
        )
    };

    let action = ExecutorAction::new(action_type, cleanup_action.map(Box::new));

    let execution_process = deployment
        .container()
        .start_execution(
            &workspace,
            &session,
            &action,
            &ExecutionProcessRunReason::CodingAgent,
        )
        .await?;

    // Clear the draft follow-up scratch on successful spawn
    // This ensures the scratch is wiped even if the user navigates away quickly
    if let Err(e) = Scratch::delete(pool, session.id, &ScratchType::DraftFollowUp).await {
        // Log but don't fail the request - scratch deletion is best-effort
        tracing::debug!(
            "Failed to delete draft follow-up scratch for session {}: {}",
            session.id,
            e
        );
    }

    Ok(ResponseJson(ApiResponse::success(execution_process)))
}

async fn load_entries_for_process(
    deployment: &DeploymentImpl,
    process: &ExecutionProcess,
) -> Result<Vec<PatchType>, ApiError> {
    let executor_action = match process.executor_action() {
        Ok(action) => action,
        Err(err) => {
            tracing::warn!(
                "Failed to parse executor action for execution {}: {}",
                process.id,
                err
            );
            return Ok(Vec::new());
        }
    };
    let is_script = matches!(executor_action.typ(), ExecutorActionType::ScriptRequest(_));

    if is_script {
        Ok(load_raw_entries(deployment, process).await)
    } else {
        Ok(load_normalized_entries(deployment, process).await?)
    }
}

async fn load_raw_entries(
    deployment: &DeploymentImpl,
    process: &ExecutionProcess,
) -> Vec<PatchType> {
    if process.status == ExecutionProcessStatus::Running {
        if let Some(store) = deployment.container().get_msg_store_by_id(&process.id).await {
            return store
                .get_history()
                .into_iter()
                .filter_map(|msg| match msg {
                    utils::log_msg::LogMsg::Stdout(content) => {
                        Some(PatchType::Stdout(content))
                    }
                    utils::log_msg::LogMsg::Stderr(content) => {
                        Some(PatchType::Stderr(content))
                    }
                    _ => None,
                })
                .collect();
        }
    }

    let log_records =
        match ExecutionProcessLogs::find_by_execution_id(&deployment.db().pool, process.id).await {
            Ok(records) => records,
            Err(err) => {
                tracing::warn!(
                    "Failed to fetch logs for execution {}: {}",
                    process.id,
                    err
                );
                return Vec::new();
            }
        };

    let raw_messages = match ExecutionProcessLogs::parse_logs(&log_records) {
        Ok(msgs) => msgs,
        Err(err) => {
            tracing::warn!(
                "Failed to parse logs for execution {}: {}",
                process.id,
                err
            );
            return Vec::new();
        }
    };

    raw_messages
        .into_iter()
        .filter_map(|msg| match msg {
            utils::log_msg::LogMsg::Stdout(content) => Some(PatchType::Stdout(content)),
            utils::log_msg::LogMsg::Stderr(content) => Some(PatchType::Stderr(content)),
            _ => None,
        })
        .collect()
}

async fn load_normalized_entries(
    deployment: &DeploymentImpl,
    process: &ExecutionProcess,
) -> Result<Vec<PatchType>, ApiError> {
    let mut doc = json!({ "entries": [] });

    if process.status == ExecutionProcessStatus::Running {
        if let Some(store) = deployment.container().get_msg_store_by_id(&process.id).await {
            let patches = store
                .get_history()
                .into_iter()
                .filter_map(|msg| match msg {
                    utils::log_msg::LogMsg::JsonPatch(patch) => Some(patch),
                    _ => None,
                });

            for patch in patches {
                if let Err(err) = apply_patch(&mut doc, &patch) {
                    tracing::warn!(
                        "Failed to apply normalized patch for execution {}: {}",
                        process.id,
                        err
                    );
                }
            }

            let entries_value = doc.get("entries").cloned().unwrap_or_default();
            return Ok(serde_json::from_value(entries_value).unwrap_or_default());
        }
    }

    let normalized_logs = match deployment
        .container()
        .get_normalized_log_messages(&process.id)
        .await
    {
        Some(logs) => logs,
        None => return Ok(Vec::new()),
    };

    for msg in normalized_logs.iter() {
        if let utils::log_msg::LogMsg::JsonPatch(patch) = msg {
            if let Err(err) = apply_patch(&mut doc, patch) {
                tracing::warn!(
                    "Failed to apply normalized patch for execution {}: {}",
                    process.id,
                    err
                );
            }
        }
    }

    let entries_value = doc.get("entries").cloned().unwrap_or_default();
    Ok(serde_json::from_value(entries_value).unwrap_or_default())
}

pub fn router(deployment: &DeploymentImpl) -> Router<DeploymentImpl> {
    let session_id_router = Router::new()
        .route("/", get(get_session))
        .route("/conversation", get(get_session_conversation))
        .route("/follow-up", post(follow_up))
        .route("/review", post(review::start_review))
        .layer(from_fn_with_state(
            deployment.clone(),
            load_session_middleware,
        ));

    let sessions_router = Router::new()
        .route("/", get(get_sessions).post(create_session))
        .nest("/{session_id}", session_id_router)
        .nest("/{session_id}/queue", queue::router(deployment));

    Router::new().nest("/sessions", sessions_router)
}
