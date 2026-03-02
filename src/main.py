import sys
from rich.markdown import Markdown
from src.query import get_vector_store, retrieve_context, generate_biohacker_response
from src.evaluator import evaluate_rag_response
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from langchain_community.chat_message_histories import ChatMessageHistory


console = Console()

history = ChatMessageHistory()


def show_welcome_screen():
    """Muestra el contexto inicial al usuario."""
    title = "[bold red]🛡️ PROTOCOLO DE SOBERANÍA PERSONAL v1.0[/bold red]"

    welcome_msg = (
        "Has ingresado al nodo de [bold]BioHacker Root[/bold].\n\n"
        "Aquí no damos consejos motivacionales. Analizamos tu [italic]biología[/italic] "
        "para que recuperes el mando de tu vida.\n\n"
        "Preguntá sobre: [yellow]Dopamina, Energía, Hardware Humano o Soberanía.[/yellow]"
    )
    console.print(Panel(welcome_msg, title=title, border_style="red", expand=False))


def main():
    show_welcome_screen()

    # 1. CARGA INICIAL (Base de Datos)
    try:
        with console.status(
            "[bold green]Cargando base de conocimiento...[/bold green]"
        ):
            db = get_vector_store("db_biohacker")
        console.print(
            "[bold green]✅ Sistema Online. Red neuronal lista.[/bold green]\n"
        )
    except Exception as e:
        console.print(
            f"[bold red]❌ Error crítico al cargar base de datos:[/bold red] {e}"
        )
        return

    while True:
        try:
            user_query = console.input(
                "[bold yellow]👤 [USUARIO] -> [/bold yellow]"
            ).strip()

            if user_query.lower() in ["salir", "exit", "quit"]:
                console.print(
                    "\n[bold red]🔌 Desconectando sistema... ¡Salí de la zona de confort, bro![/bold red]"
                )
                break

            if not user_query:
                continue

            if len(user_query.split()) < 3:
                console.print(
                    Panel(
                        "Necesito al menos [bold cyan]3 palabras[/bold cyan] para no alucinar...",
                        title="[bold yellow]⚠️ FALTA DE DATOS[/bold yellow]",
                        border_style="yellow",
                    )
                )
                continue

            # 2. PROCESAMIENTO PROTEGIDO (Retrieval + LLM + Evaluation)
            try:
                # Recuperar contexto
                console.print("[bold blue]🔍 Escaneando archivos...[/bold blue]")
                chunks = retrieve_context(user_query, db, k=3)

                # Preparar historial
                console.print(
                    "[bold magenta]🧠 Ejecutando lógica Root...[/bold magenta]"
                )
                history_str = "\n".join(
                    [f"{m.type}: {m.content}" for m in history.messages[-4:]]
                )

                answer = generate_biohacker_response(user_query, chunks, history_str)

                if "[ERROR DE SISTEMA]" in answer:
                    console.print(
                        f"\n[bold red]⚠️ Falla en red neuronal:[/bold red] {answer}"
                    )
                    continue

                history.add_user_message(user_query)
                history.add_ai_message(answer)

                console.print("\n" + "═" * 50)
                console.print(
                    Panel(
                        Markdown(answer),
                        title="[bold magenta]OUT: BIOHACKER[/bold magenta]",
                        border_style="bright_magenta",
                        padding=(1, 2),
                    )
                )
                console.print("═" * 50 + "\n")

                with console.status("[italic]Auditando calidad...[/italic]"):
                    eval_result = evaluate_rag_response(
                        user_query, answer, chunks, history_str
                    )

                final_output = {
                    "user_question": user_query,
                    "system_answer": answer,
                    "chunks_related": chunks,
                    "evaluation_bonus": eval_result,
                }
                console.print("[dim]📄 Generando log de auditoría JSON...[/dim]")
                console.print(JSON.from_data(final_output))
                console.print("\n" + "─" * 50 + "\n")

            except Exception as e:
                console.print(
                    f"\n[bold red]❌ Error de procesamiento:[/bold red] No se pudo completar la operación. {e}"
                )
                continue

        except KeyboardInterrupt:
            console.print(
                "\n\n[bold red][!] Interrupción manual. Cortaste el sistema, nos vemos.[/bold red]"
            )
            sys.exit(0)


if __name__ == "__main__":
    main()
