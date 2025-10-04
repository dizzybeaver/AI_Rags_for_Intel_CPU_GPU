@echo off
REM Copyright 2025 Joseph Hersey
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

REM Lambda RAG System Startup Script (Windows)

echo ============================================================
echo   Lambda RAG System - Starting...
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.
    pause
    exit /b 1
)

REM Check if index exists
if not exist "lambda_index_embeddings.npz" (
    echo Building initial index...
    python -c "from lambda_indexer import LambdaProjectIndexer; i = LambdaProjectIndexer(device='NPU'); i.index_project('./'); i.save_index('lambda_index')"
    echo.
)

REM Start auto-indexer in background
echo Starting auto-indexer (background)...
start /B python auto_index.py > auto_index.log 2>&1

REM Start RAG server in background
echo Starting RAG web server...
start /B python rag_server.py > rag_server.log 2>&1

REM Wait for services to start
timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo   Lambda RAG System is Ready!
echo ============================================================
echo.
echo Web UI:       http://localhost:8000
echo LM Studio:    http://localhost:1234 (make sure it's running!)
echo Auto-indexer: Running in background
echo.
echo Logs:
echo   - auto_index.log
echo   - rag_server.log
echo.
echo ============================================================
echo.
echo Press any key to stop all services...
pause >nul

REM Stop services
echo.
echo Stopping services...
taskkill /F /FI "WINDOWTITLE eq auto_index.py" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq rag_server.py" >nul 2>&1
echo Services stopped.
echo.

REM EOF
