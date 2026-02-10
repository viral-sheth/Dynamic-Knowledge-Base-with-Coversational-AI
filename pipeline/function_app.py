import os

import azure.functions as func

from .azure_integration import scheduled_cleanup_job

app = func.FunctionApp()


@app.schedule(schedule="0 0 2 * * *", arg_name="timer")
def cleanup_expired_policies(timer: func.TimerRequest):
    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    stats = scheduled_cleanup_job(conn_str)
    return func.HttpResponse(f"Cleanup completed: {stats}")
