class TaskNode():
    """Representation of a Task, compatible with Base task objects from task_scheduling"""
    def __init__(self, 
        created_time: float, 
        worker: str, 
        address: str, 
        action: str, 
        start_time: float = None,
        duration: float = None, 
        parents: list[str] = [],
    ) -> None:
        """Create the TaskNode Object with task address, worker id, parent address, along with a created timestamp."""
        self.worker = worker
        """Worker who is handling this task."""

        self.created_time = created_time
        """Time of task creation."""

        self.address = address
        """Unique identifier of Task in the form of a RuntimeAddress"""

        self.action = action
        """Type of Task ex: instansiate, sub_do_work"""

        self.start_time = start_time
        """Time of task starting. Initially 0 until found Start timestamp in logs."""

        self.started = False
        self.duration = 0.0
        """Time duration from time of start to time of finish. Initially 0 until found Finish timestamp in logs."""

        self.parents = parents
        """Identifier for direct parent object."""


    def __str__(self):
        # Optional Parent field
        return (f"{self.address} | {self.action} | {self.worker} | Created: {self.created_time} | Started: {self.start_time} | Duration: {self.duration} Parent: {self.parents}\n")
    
    def __repr__(self):
        return self.__str__()