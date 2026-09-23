def main() -> None:
    """Backward-compatible package entry point."""
    from clipper.cli import main as cli_main

    cli_main()
