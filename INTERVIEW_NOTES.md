# Notas para entrevista — lead-scoring-api

Documento interno. Não faz parte da documentação pública do projeto.

---

## As 3 decisões técnicas mais importantes

### 1. Remover `duration` — o vazamento óbvio

**O que fiz:** excluí a coluna da base e do schema da API.

**Por quê:** `duration` é a duração da ligação. Só existe depois que a ligação acabou. Uma ligação de
20 minutos quase sempre terminou em assinatura, então o modelo "aprende" o desfecho.

**O tamanho do estrago:** AUC 0,954 com ela, 0,778 sem ela (split aleatório). É a razão pela qual
esse dataset aparece reportado com 0,90+ em muito notebook por aí.

**Detalhe que vale citar:** não bastou tirar do treino. Tirei do schema Pydantic da API, então o
campo nem pode ser enviado. Se alguém mandar, é ignorado — tem teste provando que a predição não muda.

### 2. Remover as features macroeconômicas — o vazamento não óbvio

**O que fiz:** excluí `euribor3m`, `nr.employed`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`.

**Por quê, razão 1 (a que mata):** elas são iguais para todos os leads do mesmo dia. Uma feature
constante entre os itens que você está ordenando não consegue ordená-los. A tarefa é ranquear leads
entre si, não prever a taxa de conversão do trimestre.

**Por quê, razão 2 (a que mede):** a campanha atravessa a crise de 2008. No treino, `euribor3m` vai de
4,08 a 5,05. No teste, de 0,63 a 1,30. **Zero sobreposição** — 0,0% dos valores de teste aparecem no
treino. Árvore não extrapola: ela aprendeu "euribor > 4 → conversão baixa" e no teste encontra só
euribor < 1,3, fora de qualquer split que ela construiu.

**A evidência que fecha o argumento:** remover essas features **sobe** o AUC cronológico de 0,597 para
0,640 e **desce** o AUC aleatório de 0,811 para 0,778. Feature que ajuda no split aleatório e atrapalha
no split temporal é proxy de data, não sinal.

### 3. Threshold por break-even, não por 0,5

**O que fiz:** a API decide ligar quando `p ≥ custo / valor` = 8/160 = 5%.

**Por quê:** 0,5 é herança de tutorial com classe balanceada. Não tem significado de negócio. A
pergunta real é "vale 8 libras ligar para esse lead", e a resposta é a probabilidade que empata o
custo com o retorno esperado.

**O que descobri rodando:** o threshold que maximiza lucro no período de teste é 0,0 — ligar para
todo mundo. Faz sentido: a taxa base do período é 30,8% e o break-even é 5%. Todo lead é lucrativo.

Isso poderia ter virado um resultado "quebrado". Em vez disso virou o enquadramento certo: **nesse
cenário o modelo não decide quem pular, decide quem vem primeiro.** Reescrevi a avaliação em torno de
capacidade, que é a restrição real de um call center, e o número que importa passou a ser: com 30% da
capacidade, capturo 46% das conversões — 1,54x melhor que discar aleatório.

---

## 5 perguntas prováveis, com resposta

### 1. "AUC 0,64 é fraco. Por que isso é um projeto de portfólio e não um fracasso?"

Porque 0,64 é o número honesto e eu consigo mostrar exatamente de onde veio cada ponto perdido.

A sequência está no README: 0,954 com os dois vazamentos, 0,728 removendo só `duration` sob split
temporal, 0,640 removendo também os proxies de tempo. Quem reporta 0,95 nesse dataset não tem um
modelo melhor que o meu — tem um modelo que não pode ser usado.

E 0,64 não é inútil. Em lift de capacidade vira 1,54x: discando os 30% melhores, o time captura 46%
das conversões em vez de 30%. Para um call center com capacidade fixa, isso é o caso de negócio
inteiro.

A leitura que eu levaria para o gestor é essa, não o AUC: o modelo não melhora a taxa de conversão,
melhora a ordem da fila.

### 2. "O LightGBM mal ganhou da regressão logística. Por que manter a complexidade?"

Em ranqueamento, quase não ganhou mesmo: AUC 0,658 contra 0,643, e a logística até ganha no PR-AUC.
Deixei a baseline no relatório justamente porque ela quase ganha — esconder isso seria desonesto.

O que justifica é **calibração**: Brier 0,237 contra 0,264. E calibração importa aqui porque o score
alimenta um cálculo em dinheiro (`p × valor ≥ custo`), não só uma ordenação. Score mal calibrado com
boa ordenação serve para fila, não para decisão de corte.

Se o requisito fosse só ordenar, eu defenderia a logística: mais simples, mais rápida, mais fácil de
explicar para área de negócio e para compliance bancário.

### 3. "Por que split cronológico se o aleatório dá número melhor?"

Porque o número melhor é mentira sobre o que vai acontecer em produção.

O arquivo está ordenado por data de campanha. Split aleatório coloca ligações de 2010 no treino e de
2008 no teste — o modelo vê o futuro. Em produção você sempre treina no passado e pontua o futuro.

E esse dataset torna isso dramático: conversão de 4,8% no treino contra 30,8% no teste. Qualquer
avaliação que reporte uma "taxa base" única desse dataset está fazendo média de dois mundos
diferentes.

Aliás, `best_iteration` parou em 7. Isso não é bug, é sintoma: o modelo precisa ficar raso porque
profundidade não transfere através do deslocamento.

### 4. "Como a API lida com um lead de um mês que o modelo nunca viu?"

Isso acontece de verdade aqui: março, abril, setembro e dezembro caem inteiramente depois do corte de
treino. São entradas de negócio perfeitamente válidas sem nenhum histórico no modelo.

A decisão foi: o schema aceita (março é um mês real), e o `align_categories` mapeia o valor
desconhecido para NaN explicitamente. O LightGBM trata como categoria faltante e o modelo cai nas
outras features do lead. Tem teste garantindo que um lead de março retorna 200 e não 500.

O que eu **não** fiz foi deixar o pandas resolver sozinho. A versão nova emite aviso de depreciação e
a versão 4 vai levantar exceção. Tornei o comportamento explícito no código em vez de depender disso.

### 5. "Isso está pronto para produção?"

A infraestrutura sim, o modelo não — e eu separo as duas coisas.

Pronto: imagem Docker multi-stage rodando como não-root, healthcheck, validação no schema, CI que
treina, testa, constrói a imagem, sobe o container e pontua um lead através dele. Check verde
significa que a coisa realmente serve, não só que os testes passaram.

Não pronto: treinar em 2008-2009 e pontuar 2010 é exatamente o problema que esse modelo tem. Ele
precisa de retreino em janela deslizante, não de um fit único. E falta monitoramento de drift — nesse
dataset o alarme teria disparado alto.

A limitação mais séria é outra, e está no README: isso prevê **quem converte**, não **quem converte
por causa da ligação**. Parte dos leads de score alto assinaria de qualquer jeito, e ligar para eles
é custo sem receita incremental. Separar as duas coisas exige experimento, não modelo melhor.

---

## Números para ter na ponta da língua

| | |
| --- | --- |
| Contatos | 41.188 (26.360 treino / 6.590 validação / 8.238 teste) |
| AUC com os dois vazamentos (split aleatório) | 0,954 |
| AUC do modelo entregável | 0,640 |
| Conversão treino vs teste | 4,8% vs 30,8% |
| Sobreposição de `euribor3m` treino/teste | 0,0% |
| Break-even | 5% (£8 / £160) |
| Lift com 30% de capacidade | 1,54x (captura 46% das conversões) |
| Brier: LightGBM calibrado vs logística | 0,237 vs 0,264 |
| Testes | 18 |
