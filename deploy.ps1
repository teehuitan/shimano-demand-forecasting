# ============================================================
# Deployment Script - SPL Demand Forecast (scheduled batch pipeline, no web frontend)
# ============================================================
$ErrorActionPreference = "Stop"

# --- Task 1: Create .env file ---
# Matches .env.example / adj_forecast_lib/config.py - this is a scheduled batch pipeline
# (main.py), not a web app, so there is no auth/session/frontend/vector-DB config here.
$envPath = Join-Path "$env:bamboo_path" ".env"
$envContent = @(
    "ODBC_DRIVER=$env:bamboo_ODBC_DRIVER"

    # QA SQL Server (adj_forecast_lib/db_writer.py, lifecycle_cleaning.py, wipe_test_run.py)
    "QA_DB_SERVER=$env:bamboo_QA_DB_SERVER"
    "QA_DB_UID=$env:bamboo_QA_DB_UID"
    "QA_DB_PWD=$env:bamboo_QA_DB_PWD"

    # Fabric SQL Server (adj_forecast_lib/preprocessing.py, collate_outstanding_order.py)
    "FABRIC_DB_SERVER=$env:bamboo_FABRIC_DB_SERVER"
    "FABRIC_DB_UID=$env:bamboo_FABRIC_DB_UID"
    "FABRIC_DB_PWD=$env:bamboo_FABRIC_DB_PWD"

    # Pipeline config (adj_forecast_lib/config.py) — required at import time, no defaults
    "IO_SQL_PATH=$env:bamboo_IO_SQL_PATH"
    "IO_OUTSTANDING_SQL_PATH=$env:bamboo_IO_OUTSTANDING_SQL_PATH"
    "N_ESTIMATORS=$env:bamboo_N_ESTIMATORS"
    "EARLY_STOPPING_ROUNDS=$env:bamboo_EARLY_STOPPING_ROUNDS"
    "PLANT_NAMES=$env:bamboo_PLANT_NAMES"
    "PLANT_EXCLUDE_PARTCODES=$env:bamboo_PLANT_EXCLUDE_PARTCODES"
    "STJ_EXTERNAL_EVENT_RANGES=$env:bamboo_STJ_EXTERNAL_EVENT_RANGES"
    "SPL_EXTERNAL_EVENT_RANGES=$env:bamboo_SPL_EXTERNAL_EVENT_RANGES"
    "STJ_EXTERNAL_EVENT_END_KNOWN=$env:bamboo_STJ_EXTERNAL_EVENT_END_KNOWN"
    "SPL_EXTERNAL_EVENT_END_KNOWN=$env:bamboo_SPL_EXTERNAL_EVENT_END_KNOWN"
    "PLANT_INCLUDES_CNY=$env:bamboo_PLANT_INCLUDES_CNY"
    "PLANT_INCLUDES_PRED_SUCC=$env:bamboo_PLANT_INCLUDES_PRED_SUCC"
    "PLANT_EXCLUDES_SOLD_FROM_OUTSTANDING=$env:bamboo_PLANT_EXCLUDES_SOLD_FROM_OUTSTANDING"
    "PLANT_INCLUDES_MANUAL_METHOD=$env:bamboo_PLANT_INCLUDES_MANUAL_METHOD"
    "COVID_YYYYMM_RANGE=$env:bamboo_COVID_YYYYMM_RANGE"
    "PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES=$env:bamboo_PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES"
    "MLFLOW_TRACKING_URI=$env:bamboo_MLFLOW_TRACKING_URI"

    # main.py daily orchestrator
    "DAILY_PLANTS=$env:bamboo_DAILY_PLANTS"
    "DAILY_LABEL=$env:bamboo_DAILY_LABEL"
    "DAILY_COVID_WEIGHT=$env:bamboo_DAILY_COVID_WEIGHT"
    "DAILY_VAL_MONTHS=$env:bamboo_DAILY_VAL_MONTHS"
    "DAILY_HORIZONS=$env:bamboo_DAILY_HORIZONS"
    "DAILY_MIN_ORIGIN_MONTHS_SINCE_LAUNCH=$env:bamboo_DAILY_MIN_ORIGIN_MONTHS_SINCE_LAUNCH"
    "DAILY_MANUAL_WINDOW=$env:bamboo_DAILY_MANUAL_WINDOW"
    "DAILY_CUTOFF_GRAIN=$env:bamboo_DAILY_CUTOFF_GRAIN"
    "DAILY_FORCE_RUN=$env:bamboo_DAILY_FORCE_RUN"
    "DAILY_FORCE_RETRAIN=$env:bamboo_DAILY_FORCE_RETRAIN"
    ""
    # Log mirroring into [AI].AI.AIAPPLOGS (adj_forecast_lib/db_logging.py)
    "LOG_DB_ENABLED=$env:bamboo_LOG_DB_ENABLED"
    "LOG_DB_ENVIRONMENT=$env:bamboo_LOG_DB_ENVIRONMENT"
    "LOG_RETENTION_MONTHS=$env:bamboo_LOG_RETENTION_MONTHS"
)
$envContent | Set-Content $envPath

# --- Task 2: Create run.bat - what the Scheduled Task actually executes ---
# Just runs the batch pipeline directly (`python main.py`). No uvicorn/ASGI app, no server
# process to keep alive - main.py does its work once and exits.
$runBatPath = Join-Path "$env:bamboo_path" "run.bat"
$runBatLocalPath = Join-Path "$env:bamboo_Local_Path" "run.bat"
$runBatContent = @(
    "@echo off",
    "cd /d `"$env:bamboo_Local_Path`"",
    "uv run python main.py"
)
$runBatContent | Set-Content $runBatPath

# --- Task 3: Create start.bat - deploy-time setup, run once via SSH below ---
$path = "$env:bamboo_Local_Path"
$startBatPath = Join-Path "$env:bamboo_path" "start.bat"

$startBatContent = @(
    "@echo on",
    "cd /d `"$path`"",
    "",
    "REM === Setup Python Backend (uv) ===",
    "REM Verify uv is on PATH for the deployment account before relying on it",
    "where uv >nul 2>nul",
    "if %ERRORLEVEL% NEQ 0 (",
    "    echo ERROR: uv not found on PATH - install it for $env:bamboo_schtask_user",
    "    exit /b 1",
    ")",
    "",
    "REM A failed sync leaves a .venv whose interpreter does not work, and uv will",
    "REM not repair it - it just fails with 'Querying Python at ...\\.venv\\Scripts\\",
    "REM python.exe failed'. .venv is not in the artifact, so it survives every",
    "REM deploy. Probe it and drop it only when broken, so a healthy env is still",
    "REM reused and uv sync stays a no-op.",
    "if not exist `".venv\Scripts\python.exe`" goto venv_ok",
    "`".venv\Scripts\python.exe`" -c `"pass`" >nul 2>nul",
    "if not errorlevel 1 goto venv_ok",
    "echo Stale .venv found - removing so uv can rebuild it",
    "rmdir /s /q `".venv`"",
    ":venv_ok",
    "",
    "REM uv sync creates .venv, fetches the interpreter pinned in .python-version,",
    "REM and installs the exact versions in uv.lock. It is idempotent, so there is",
    "REM no hash check to short-circuit - an in-sync env is already a no-op.",
    "REM --locked asserts uv.lock is current for pyproject.toml and fails if it is",
    "REM not, so a dependency change committed without relocking cannot deploy.",
    "uv sync --locked",
    "if %ERRORLEVEL% NEQ 0 (",
    "    echo ERROR: uv sync --locked failed - uv.lock is stale or a package would not install",
    "    echo Run 'uv lock' locally and commit uv.lock alongside pyproject.toml",
    "    exit /b 1",
    ")",
    "echo uv sync completed",
    "",
    "REM === Schedule daily 11:00 start ===",
    "echo Configuring Scheduled Task...",
    "schtasks /Delete /TN `"$env:bamboo_schtask_folder\$env:bamboo_project_name`" /F 2>nul",
    "schtasks /Create /TN `"$env:bamboo_schtask_folder\$env:bamboo_project_name`" /TR `"\`"$runBatLocalPath\`"`" /SC DAILY /ST 11:00 /RU $env:bamboo_schtask_user /RL HIGHEST /F",
    "",
    "REM Add ONSTART trigger and remove execution time limit via PowerShell",
    "powershell -Command `"& { `$task = Get-ScheduledTask -TaskPath '\$env:bamboo_schtask_folder\' -TaskName '$env:bamboo_project_name'; `$startupTrigger = New-ScheduledTaskTrigger -AtStartup -RandomDelay (New-TimeSpan -Seconds $env:bamboo_startup_delay_seconds); `$task.Triggers += `$startupTrigger; `$task.Settings.ExecutionTimeLimit = 'PT0S'; Set-ScheduledTask -TaskName '$env:bamboo_project_name' -TaskPath '\$env:bamboo_schtask_folder\' -Trigger `$task.Triggers -Settings `$task.Settings }`"",
    "",
    "echo Scheduled task configured: daily at 11:00, plus on system startup",
    "",
    "REM === Run the pipeline once now, so a deploy doesn't have to wait for the next 11:00 ===",
    "REM trigger or a reboot. Uses run.bat directly (not schtasks /Run) so this call blocks until",
    "REM main.py finishes and its exit code propagates to the deploy.",
    "echo Running pipeline once immediately...",
    "call `"$runBatLocalPath`"",
    "if %ERRORLEVEL% NEQ 0 (",
    "    echo ERROR: immediate pipeline run failed with exit code %ERRORLEVEL%",
    "    exit /b 1",
    ")",
    "echo Immediate pipeline run completed",
    "",
    "echo Deployment completed successfully",
    "exit /b 0"
)
$startBatContent | Set-Content $startBatPath

# --- Task 4: Trigger start.bat via SSH ---
Write-Host "Connecting to remote server $env:bamboo_ssh_server..."
ssh -o StrictHostKeyChecking=no "$env:bamboo_ssh_account@$env:bamboo_ssh_server" "`"$startBatPath`""
if ($LASTEXITCODE -ne 0) {
    throw "SSH remote execution failed with exit code $LASTEXITCODE"
}

Write-Host "Deployment completed successfully."