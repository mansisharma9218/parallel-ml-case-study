"""The shared raw-results CSV schema every sub-problem must write to, so
rows from S01/S02/S03 merge cleanly into the class master dataset.
"""

COLUMNS = [
    "Student_ID",
    "Batch_ID",
    "Workload",
    "Dataset",
    "Problem_Size",
    "Algorithm",
    "Parallel_Model",
    "Hardware",
    "Resource_Count",
    "Threads",
    "Execution_Time",
    "Speedup",
    "Efficiency",
    "Communication_Time",
    "Synchronization_Time",
    "Scheduling_Time",
    "Memory_Use",
    "Energy_if_available",
    "Run_ID",
]
