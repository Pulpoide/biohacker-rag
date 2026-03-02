import shutil
import os
from rich.console import Console

console = Console()


def reset_database():
    db_path = "db_biohacker"

    if os.path.exists(db_path):
        try:
            console.print(
                f"[bold yellow]⚠️  Eliminando base de datos vectorial en: {db_path}...[/bold yellow]"
            )
            shutil.rmtree(db_path)
            console.print(
                "[bold green]✅ Base de datos eliminada con éxito.[/bold green]"
            )
        except Exception as e:
            console.print(f"[bold red]❌ Error al eliminar la DB: {e}[/bold red]")
    else:
        console.print(
            "[blue]ℹ️  No existe una base de datos previa. El sistema está limpio.[/blue]"
        )

    console.print(
        "\n[bold cyan]🚀 Para reconstruir el cerebro de Root, ejecutá:[/bold cyan]"
    )
    console.print("python src/build_index.py")


if __name__ == "__main__":
    reset_database()
