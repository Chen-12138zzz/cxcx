@echo off
REM 本机 slidep 包装脚本依赖 coreutils，PATH 残缺会失败；这里直接调入口。
REM 用法: sp.cmd lint <pptx> <slide>
REM       sp.cmd upsert <pptx> <slide> [page-index]
set NODE=C:\Users\Chen\.workbuddy\binaries\node\versions\22.22.2-3\node.exe
set ENTRY=C:\Users\Chen\.workbuddy\binaries\node\versions\22.22.2-3\node_modules\@tencent\slidep\dist\index.js

if /I "%1"=="lint" (
  "%NODE%" "%ENTRY%" lint %2 --dsl-file %3
  exit /b %ERRORLEVEL%
)
if /I "%1"=="upsert" (
  if "%4"=="" (
    "%NODE%" "%ENTRY%" upsert-dsl %2 --dsl-file %3
  ) else (
    "%NODE%" "%ENTRY%" upsert-dsl %2 --dsl-file %3 --page-index %4
  )
  exit /b %ERRORLEVEL%
)
echo usage: sp.cmd lint^|upsert ^<pptx^> ^<slide^> [page-index]
exit /b 2
