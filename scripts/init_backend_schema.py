from grade_system.persistence import init_database_schema


def main() -> None:
    init_database_schema()
    print("Database schema initialized.")


if __name__ == "__main__":
    main()
