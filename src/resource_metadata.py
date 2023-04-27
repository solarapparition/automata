import sqlite3
from typing import Optional, Tuple


class ResourceMetadata:
    """
    A class for managing resource metadata in an SQLite database.

    Args:
        db_path (str): Path to the SQLite database file. Default is "db/resource_metadata.db".
    """

    def __init__(self, db_path: str = "db/resource_metadata.db") -> None:
        """
        Initializes a new instance of the ResourceMetadata class.

        Args:
            db_path (str): Path to the SQLite database file. Default is "db/resource_metadata.db".
        """
        self.conn: sqlite3.Connection = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self) -> None:
        """
        Creates a new table in the database if it does not already exist.
        """
        cursor: sqlite3.Cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                description TEXT
            )
        """
        )
        self.conn.commit()

    def set_description(self, resource_id: str, description: str) -> None:
        """
        Sets the description for a resource in the database.

        Args:
            resource_id (str): The ID of the resource.
            description (str): The description of the resource.
        """
        cursor: sqlite3.Cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO metadata (id, description)
            VALUES (?, ?)
        """,
            (resource_id, description),
        )
        self.conn.commit()

    def delete_resource(self, resource_id: str) -> None:
        """
        Deletes a resource from the database.

        Args:
            resource_id (str): The ID of the resource to delete.
        """
        cursor: sqlite3.Cursor = self.conn.cursor()
        cursor.execute(
            """
            DELETE FROM metadata WHERE id=?
        """,
            (resource_id,),
        )
        self.conn.commit()

    def get_description(self, resource_id: str) -> Optional[str]:
        """
        Gets the description for a resource from the database.

        Args:
            resource_id (str): The ID of the resource to retrieve.

        Returns:
            Optional[str]: The description of the resource, or None if the resource does not exist in the database.
        """
        cursor: sqlite3.Cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT description FROM metadata WHERE id=?
        """,
            (resource_id,),
        )
        result: Optional[Tuple[str]] = cursor.fetchone()
        return result[0] if result else None


def demo() -> None:
    """
    A demo of the ResourceMetadata class.
    """
    metadata_tracker = ResourceMetadata()

    metadata_tracker.set_description("resource_1", "This is resource 1")
    print(metadata_tracker.get_description("resource_1"))

    metadata_tracker.delete_resource("resource_1")
    print(metadata_tracker.get_description("resource_1"))


if __name__ == "__main__":
    demo()
