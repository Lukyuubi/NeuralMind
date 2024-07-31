# NeuralMind
##Introdução
Este relatório visa o informar a trajetória e o contato inicial de um chatbot para responder a perguntas de vestibular usando processamento de linguagem natural (NLP) e recuperação de informações de documentos PDF. O código utiliza várias bibliotecas, incluindo streamlit para a interface do usuário, langchain para o pipeline de processamento de documentos e recuperação, e OpenAI para geração de embeddings e Llama3 como llm para respostas.

##Visão Geral
O código utiliza o conceito de RAG  e possui dois componentes principais: Indexing e Retrieval and Generation.
###Indexing
A seção de indexing utiliza a biblioteca PyPDFLoader, Chroma, e OpenAIEmbeddings. Essas bibliotecas são responsáveis pelo carregamento dos dados, divisão do texto, armazenamento dos vetores e geração dos embeddings. A principal funcionalidade do split é quebrar o documento em pedaços menores chamados de chunks, o que permite vetorizar (embeddings) e armazenar (ChromaDB) trechos do documento para serem utilizados na geração das respostas pela LLM.

###Retrieval and Generation
Para o componente de Retrieval and Generation, o código utiliza um recuperador de documentos (retriever), um prompt de sistema e o modelo de linguagem (LLM) Llama3-70b-8192. Dada uma entrada, a função retriever é responsável por devolver splits relevantes do banco de dados para a geração da resposta, com a LLM produzindo uma resposta contextualizada com base no prompt descrito.

##Parâmetros de busca
Para a divisão do documento PDF em chunks, foram utilizados tamanhos de 128 tokens com overlap de 10 tokens (chunk_size=128, chunk_overlap=10). A justificativa para essa escolha é que chunks menores ajudam a manter a coerência e o contexto das perguntas feitas, permitindo um contexto mais focado e relevante para geração de respostas, além de uma busca mais precisa.
A função retriever utilizada no banco de dados Chroma busca por similaridade **(search_type="similarity")** e retorna os 5 documentos mais relevantes para cada consulta **(search_kwargs={'k': 5}).** Foram realizados testes manuais com outros tipos de busca, como similarity_score_threshold e maximum marginal relevance retrieval, mas os melhores resultados foram obtidos com a busca por similaridade. Uma hipótese para isso é a proximidade semântica entre as perguntas feitas e os chunks obtidos pela divisão do documento.

##Resultados
Os testes realizados foram feitos de maneira interativa com o código rodando localmente. Perguntas foram refeitas e parâmetros de busca alterados para obter respostas melhores sobre um mesmo contexto. Apesar de tentativas de mudança na LLM, o principal problema enfrentado foi a falta de treinamento para respostas erradas. Não obtive sucesso em realizar um treinamento com as perguntas incorretas.

Outro problema encontrado foi a leitura incompleta de dados de tabelas. Devido ao tamanho dos chunks, as respostas geradas frequentemente não continham informações detalhadas ou o chatbot afirmava que não possuía conhecimento sobre o assunto. Por exemplo, ao perguntar sobre a quantidade de vagas para um determinado curso, a resposta era frequentemente "Não há informações específicas sobre vagas para engenharia de produção no texto fornecido", devido à leitura incompleta das tabelas que não conseguiam assimilar as colunas e os conteúdos numéricos corretamente.

##Conclusões
O desenvolvimento do chatbot apresentou desafios significativos, especialmente na extração de informações detalhadas de documentos PDF complexos. A divisão em chunks menores foi eficaz para manter a coerência e relevância do contexto, mas a falta de um sistema robusto de feedback e treinamento de respostas incorretas limitou a precisão do chatbot. Problemas específicos na leitura de tabelas indicam a necessidade de uma abordagem mais sofisticada para a extração de dados tabulares.

##Visões futuras e Melhorias
Para melhorar o sistema, proponho as seguintes abordagens:

**Sistema de Feedback:** Implementar um sistema de feedback onde os usuários possam indicar se a resposta foi correta ou não, armazenando esse feedback para uso posterior.
**Treinamento com Feedback:** Utilizar o feedback coletado para re-treinar o modelo, melhorando sua capacidade de responder corretamente a perguntas semelhantes no futuro.
**Extração de Dados de Tabelas:** Desenvolver métodos mais avançados para a extração e interpretação de dados tabulares, possivelmente utilizando técnicas de OCR e processamento de tabelas.
**Interface de Usuário:** Melhorar a interface do usuário para tornar a interação mais intuitiva e amigável, possivelmente utilizando Streamlit para criar uma experiência mais interativa e visualmente agradável.
**Exportação de Dados:** Considerar a exportação de dados de treinamento e respostas corretas em formatos como JSON, permitindo um aprendizado contínuo e melhorias iterativas.

#Arquivos
O arquivo **local-chatbot.py** foi o arquivo inicial criado para testagem local via terminal, enquanto o arquivo app.py busca utilizar a interface do streamlit para hospedagem.
Para rodar o arquivo app.py localmente utilize o comando **streamlit run app.py**
