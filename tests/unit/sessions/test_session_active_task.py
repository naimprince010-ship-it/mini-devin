from mini_devin.sessions.manager import Session, SessionStatus, Task, TaskStatus


def test_active_task_id_repairs_missing_pointer() -> None:
    session = Session(
        session_id="s1",
        working_directory="/tmp",
        model="gpt-4.1",
        max_iterations=10,
        status=SessionStatus.RUNNING,
    )
    session.tasks["t1"] = Task(
        task_id="t1", description="run", status=TaskStatus.RUNNING
    )

    assert session.active_task_id() == "t1"
    assert session.current_task_id == "t1"


def test_active_task_id_clears_stale_pointer() -> None:
    session = Session(
        session_id="s1",
        working_directory="/tmp",
        model="gpt-4.1",
        max_iterations=10,
        status=SessionStatus.RUNNING,
        current_task_id="missing",
    )

    assert session.active_task_id() is None
    assert session.current_task_id is None
