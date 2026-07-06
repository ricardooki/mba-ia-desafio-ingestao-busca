from search import answer_question, search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("=== Chat de Busca ===")
    print("Digite sua pergunta. Para sair, digite 'sair', 'exit' ou 'quit'.")

    while True:
        question = input("Pergunta> ").strip()
        if not question:
            continue
        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            break

        try:
            answer = answer_question(question, chain)
            print(f"\nResposta: {answer}\n")
        except Exception as exc:
            print(f"Erro ao processar a pergunta: {exc}")
            break


if __name__ == "__main__":
    main()
