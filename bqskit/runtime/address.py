"""This module implements the RuntimeAddress NamedTuple."""
from __future__ import annotations

from typing import NamedTuple


class RuntimeAddress(NamedTuple):
    """
    An address to a mailbox on a server or worker.

    These are also used as unique identification for a task, since in this
    system, a mailbox slot has a one-to-one relationship with a task. A task
    will ship its result to its return address. That slot will only accept a
    package from that task.
    """
    worker_id: int
    mailbox_index: int
    mailbox_slot: int

    def __str__(self):
        return (f"{self.worker_id}:{self.mailbox_index}:{self.mailbox_slot}")
    
    def __repr__(self):
        return self.__str__()
