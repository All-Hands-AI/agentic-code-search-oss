import pytest

from src.utils.dataset import extract_functions_from_patch


def test_extract_functions_from_patch():
    diff = """diff --git a/moto/rds/exceptions.py b/moto/rds/exceptions.py
--- a/moto/rds/exceptions.py
+++ b/moto/rds/exceptions.py
@@ -82,6 +82,14 @@ def __init__(self, database_identifier: str):
)


+class DBClusterToBeDeletedHasActiveMembers(RDSClientError):
+ def __init__(self) -> None:
+ super().__init__(
+ "InvalidDBClusterStateFault",
+ "Cluster cannot be deleted, it still contains DB instances in non-deleting state.",
+ )
+
+
class InvalidDBInstanceStateError(RDSClientError):
def __init__(self, database_identifier: str, istate: str):
estate = (
diff --git a/moto/rds/models.py b/moto/rds/models.py
--- a/moto/rds/models.py
+++ b/moto/rds/models.py
@@ -19,6 +19,7 @@
DBClusterNotFoundError,
DBClusterSnapshotAlreadyExistsError,
DBClusterSnapshotNotFoundError,
+ DBClusterToBeDeletedHasActiveMembers,
DBInstanceNotFoundError,
DBSnapshotNotFoundError,
DBSecurityGroupNotFoundError,
@@ -2339,7 +2340,8 @@ def delete_db_cluster(
raise InvalidParameterValue(
"Can't delete Cluster with protection enabled"
)
-
+ if cluster.cluster_members:
+ raise DBClusterToBeDeletedHasActiveMembers()
global_id = cluster.global_cluster_identifier or ""
if global_id in self.global_clusters:
self.remove_from_global_cluster(global_id, cluster_identifier)"""

    result = extract_functions_from_patch(diff)

    # Note: In unified diff headers, paths are prefixed with a/ and b/.
    # After stripping the leading 'b/', the actual path here is 'b/b.py'.
    assert result == [
        ("moto/rds/exceptions.py", [82, 6]),
        ("moto/rds/models.py", [19, 6]),
        ("moto/rds/models.py", [2339, 7])
    ]