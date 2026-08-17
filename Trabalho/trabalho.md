# L-SHADE GRADUADA POR DISTÂNCIA COM DEFLAÇÃO DE BACIA

**Trabalho final — CPC 881, Métodos Computacionais Inspirados na Natureza (2026.2)**

Documento de projeto: o que foi medido, o que foi mudado, por quê, o que os números dizem e como a
proposta se posiciona diante da literatura recente.

| onde | o quê |
| --- | --- |
| `Nature/Models/AdaptiveDE/index.py` | o algoritmo — quarta variante (`variant='graded'`) da classe que já hospeda JADE, SHADE e L-SHADE |
| `Nature/index.py` | `NatureSelector('graded_de', …)`, a fachada que o notebook usa |
| `Trabalho/Analysis.ipynb` | o experimento — 12 campanhas, 6 algoritmos, 30 corridas cada |
| `Trabalho/trabalho.md` | este documento |

### Índice

1. [Método de referência](#1-método-de-referência-l-shade)
2. [Diagnóstico — onde a L-SHADE perde](#2-diagnóstico--onde-a-l-shade-perde-e-por-quê)
3. [A proposta, explicada do zero](#3-a-proposta-explicada-do-zero)
4. [Estado da arte e ineditismo](#4-estado-da-arte-e-ineditismo)
5. [Resultados](#5-resultados)
6. [Mapa dos requisitos do enunciado](#6-mapa-dos-requisitos-do-enunciado)
7. [Limitações](#7-limitações)
8. [Como reproduzir](#8-como-reproduzir)
9. [Referências](#9-referências)

### Resumo em dez linhas

A L-SHADE, vencedora do CEC'14, falha de duas maneiras distintas nas seis funções do enunciado. Na **F4 em
$D{=}30$ ela termina em $76{,}5754258$ nas 30 corridas, com desvio-padrão exatamente zero** — sempre a mesma
bacia errada, e quadruplicar o orçamento não muda nada. Na **F9 e na F13 ela nunca chega ao alvo** por um
motivo oposto: a população continua viva até a última avaliação, mas anda pouco — o passo de cada indivíduo
é proporcional à distância dele ao atrator, então o miolo da nuvem quase não se move. A proposta ataca uma
falha com cada mecanismo: (1) o **passo de atração de comprimento
uniforme**, que troca o fator de escala único da DE por um fator por indivíduo governado pela distância
euclidiana ao incumbente, e (2) a **deflação de bacia**, que proíbe a região já exaurida quando a busca
comprova que está desperdiçando orçamento. Resultado em 30 corridas: **F4 $D{=}30$ de $76{,}575$ para
$4{,}569$ de mediana** (de 0/30 para 13/30 corridas que escapam da bacia, $p < 10^{-4}$), **F9 de $15{,}86$
para $12{,}45$ em $D{=}30$ e de $23{,}50$ para $17{,}85$ em $D{=}50$** ($p < 10^{-4}$), e **nenhum dano** nas
duas funções que a original já resolvia (F2 e F7 seguem 30/30 nas duas dimensões).

---

## 1. MÉTODO DE REFERÊNCIA: L-SHADE

O método escolhido é a **L-SHADE** (Tanabe & Fukunaga, CEC 2014), vencedora da competição CEC'14 — a mesma
suíte que o enunciado manda otimizar. Ela é a mais forte das cinco implementações do pacote `Nature`: tem a
melhor mediana na grande maioria das doze campanhas. Modificar a mais fraca seria fácil e não diria nada; a
pergunta que interessa é onde a melhor ainda perde.

A L-SHADE é uma DE `current-to-pbest/1/bin` com arquivo, memória de sucesso e população decrescente. A
equação de mutação, que é onde a proposta mexe, é:

$$v_i \;=\; x_i \;+\; \underbrace{F_i\,(x_{p\text{best}} - x_i)}_{\text{atração}} \;+\; \underbrace{F_i\,(x_{r_1} - \tilde{x}_{r_2})}_{\text{difusão}}$$

- **Atração**: puxa o indivíduo $x_i$ na direção de $x_{p\text{best}}$, sorteado entre os $p\%$ melhores
  ($p = 0{,}11$). É o termo que faz a população convergir.
- **Difusão**: soma a diferença entre dois indivíduos sorteados, com $\tilde{x}_{r_2}$ vindo de
  população ∪ arquivo (arquivo com $2{,}6N$ vagas). É o termo que explora.
- $F_i \sim \text{Cauchy}(M_{F,r_i},\,0{,}1)$, **saturado em 1**, e $CR_i \sim \mathcal{N}(M_{CR,r_i},\,0{,}1)$.
  As memórias $M_F$ e $M_{CR}$ ($H = 6$ células, circulares) são atualizadas pela média de Lehmer ponderada
  pelo **ganho de aptidão** dos ensaios que venceram a seleção.
- **LPSR**: a população cai linearmente de $18D$ até 4 ao longo do orçamento,
  $N(t) = \text{round}\!\left[(4 - N_{\text{init}})\,t + N_{\text{init}}\right]$, com $t = \text{FES}/\text{MaxFES}$.

**O detalhe que importa para tudo o que vem depois:** o mesmo $F_i$ multiplica os dois termos, e o termo de
atração é proporcional a $(x_{p\text{best}} - x_i)$. Ou seja, **o comprimento do passo de cada indivíduo é
proporcional à distância dele ao atrator**.

---

## 2. DIAGNÓSTICO — ONDE A L-SHADE PERDE, E POR QUÊ

Nada foi proposto antes de medir. Todas as medições usam o protocolo do enunciado: MaxFES $= 10^4 D$, caixa
$[-100,100]^D$, inicialização não-determinística, parada em erro $< 10^{-8}$.

### 2.1 Em uma das campanhas, 78% do orçamento é gasto com a população morta

Traçando erro, diversidade e taxa de sucesso ao longo de uma corrida em $D = 30$:

| campanha | erro final | erro chega a +10% do final em | espalhamento no fim | sucesso na metade do ciclo |
| --- | --- | --- | --- | --- |
| F2 bent cigar | $10^{-8}$ (alvo) | 66% | 2,9e−06 | 33% |
| **F4 rosenbrock** | **76,575** | **13% do orçamento** | **5e−08** | **0,5%** |
| F6 weierstrass | $10^{-8}$ (alvo) | 68% | 29,9 | 9,7% |
| F7 griewank | $10^{-8}$ (alvo) | 37% | 55,7 | 71% |
| F9 rastrigin | 11,9 | 98% | 19,0 | 5,0% |
| F13 happycat | 0,191 | 88% | 14,5 | 0,9% |

A F4 é o caso extremo: **o erro final já está fixado em 22% do orçamento** e o desvio médio por coordenada
cai para $5\times10^{-8}$ — a nuvem inteira ocupa menos de um bilionésimo da aresta da caixa. As 78%
restantes das 300 mil avaliações produzem ensaios que diferem do pai na oitava casa decimal. Nas F9 e F13,
ao contrário, a população ainda está viva e melhorando na última avaliação: ali **não há desperdício**, e
qualquer mecanismo de religamento tem de ficar quieto.

### 2.2 A F4 é o mesmo ponto errado em 30 de 30 corridas

Na F4 em $D = 30$, as 30 corridas terminam em $76{,}5754258$ — **com desvio-padrão exatamente zero**. Não é
acaso e não é falta de orçamento:

- rodar com **4× o teto de avaliações** (fora do enunciado, só para diagnóstico) devolve o mesmo
  $76{,}5754258$;
- o ponto tem $\lVert\nabla f\rVert = 2{,}49$ e **três coordenadas grudadas em $+100$**: é um ponto KKT da
  caixa, não um mínimo interior. Um L-BFGS-B partindo dali não sai do lugar;
- um multistart de L-BFGS-B com 30 partidas uniformes cai em **seis bacias distintas** —
  `{0,0: 4 vezes; 3,99: 5; 56,8: 1; 76,575: 13; 94,1: 1; 113,3: 6}`. A bacia de $76{,}575$ é a mais larga do
  relevo, e o **ótimo global é alcançável por descida local pura em 4 das 30 partidas**;
- o ótimo global está a $\lVert x^\* - o\rVert = 230$ do ponto de colapso (a diagonal da caixa é 1095).

A leitura é dura: **a busca populacional acha o global 0 vezes em 30, onde o gradiente puro acha 4 vezes em
30.** A população grande recomendada ($N = 18D = 540$) *piora* isso — ela promedia o relevo e escorrega
sempre para o funil mais largo. Com $N = 100$, ou com SHADE/JADE sem LPSR, 1 a 2 corridas em 15 escapam.

### 2.3 Religar não resolve — só proibir resolve

Como todas as corridas independentes caem na mesma bacia, **reinicializar uniformemente é inútil**: a nuvem
nova reencontra o mesmo funil. Medido (§5.2): religar na estagnação, mantendo o melhor de todos, devolve
$76{,}5754258$ em 30 de 30 corridas — exatamente o resultado de não fazer nada. O que muda o jogo é
*proibir* a bacia já exaurida.

### 2.4 O passo da DE morre junto com a diversidade

O termo de atração $F(x_{p\text{best}} - x_i)$ tem comprimento **proporcional à distância ao atrator**.
Consequência: cada indivíduo anda sempre a *mesma fração* do que falta, a nuvem contrai de forma
auto-semelhante, e o passo encolhe junto com o que resta de diversidade. **A DE não tem escala própria** —
ela só sabe gerar deslocamentos do tamanho da nuvem. É a causa estrutural do que a §2.1 mede: quando a
diversidade morre, o algoritmo não tem como voltar a dar passos grandes, mesmo com metade do orçamento na
mão.

### 2.5 Um desvio do código original (corrigido nos dois braços)

O tratamento de borda da `AdaptiveDE` era `np.clip`. O código publicado da L-SHADE usa o **reparo pelo ponto
médio** — o gene fora da caixa vai para o meio entre o pai e a borda:

```python
v[low]  = (lb[low]  + x_i[low])  / 2.0
v[high] = (ub[high] + x_i[high]) / 2.0
```

O `clip` empilha a população sobre a face da caixa e **congela a coordenada**, porque a diferença entre dois
indivíduos grudados no mesmo limite é zero — e a F4 termina justamente com três coordenadas em $+100$. A
correção foi aplicada e, por ser correção de fidelidade ao método de referência, **vale para os dois braços
da comparação**: o `adaptive_de` do notebook também passou a usá-la. Ela sozinha melhora a F6 ($D{=}30$:
mediana 2,0 → 0) e **não** resolve a F4. Ela não é contribuição — é conserto.

---

## 3. A PROPOSTA, EXPLICADA DO ZERO

Duas modificações, uma para cada falha diagnosticada. As duas vivem na variante `graded`; com
`variant='lshade'` o mesmo arquivo roda a L-SHADE fiel, linha por linha.

### 3.1 Parte 1 — passo de atração de comprimento uniforme

#### 3.1.1 O problema, em uma frase

Na L-SHADE, **quem está longe do incumbente dá passos enormes e é engolido num lance; quem está perto dá
passos minúsculos e não sai do lugar.** As duas coisas são o mesmo defeito: o passo é proporcional à
distância.

#### 3.1.2 Como era

$$v_i = x_i + \underbrace{F_i}_{\text{um só fator}}(x_{p\text{best}} - x_i) + F_i\,(x_{r_1} - \tilde{x}_{r_2})$$

O comprimento do passo de atração é

$$\text{passo}_i = F_i \cdot d_i, \qquad d_i = \lVert x_i - x_{\text{atrator}}\rVert$$

#### 3.1.3 Como ficou

O termo de atração ganha um fator de escala **próprio de cada indivíduo**, governado pela distância
euclidiana dele ao incumbente:

$$\boxed{\;v_i = x_i + F_i^{\text{pull}}\,(x_{p\text{best}} - x_i) + F_i\,(x_{r_1} - \tilde{x}_{r_2})\;}$$

$$\boxed{\;F_i^{\text{pull}} \;=\; \min\!\left(1,\;\; F_i \left(\frac{d_{\max}}{d_i}\right)^{(1-t)^2}\right)\;}$$

$$d_i = \lVert x_i - x_{\text{best}}\rVert, \qquad d_{\max} = \max_j d_j, \qquad t = \frac{\text{FES}}{\text{MaxFES}}$$

O termo de difusão **não muda** — continua com o $F_i$ original. A memória de sucesso **não muda** —
continua aprendendo $M_F$ pelo ganho de aptidão, como no SHADE original.

#### 3.1.4 O que isso faz, em português

No começo do ciclo ($t \approx 0$, expoente $= 1$), a expressão colapsa em algo muito simples. Como
$\text{passo}_i = F_i^{\text{pull}}\cdot d_i$:

$$\text{passo}_i \;=\; \min\big(\,d_i,\;\; F_i \cdot d_{\max}\,\big)$$

> **"Ande $F\cdot d_{\max}$ na direção do atrator. Se você já estiver mais perto do que isso, ande até ele."**

$d_{\max}$ é o **raio da nuvem** — a distância do indivíduo mais afastado até o incumbente. Todo mundo passa
a andar *o mesmo comprimento*, em vez de uma fração da própria distância.

#### 3.1.5 Exemplo numérico

Cinco indivíduos a distâncias 100, 50, 20, 5 e 1 do incumbente, com $F = 0{,}5$ e $d_{\max} = 100$:

| $d_i$ | L-SHADE: passo $= F d_i$ | proposta $t{=}0$ | proposta $t{=}0{,}5$ | proposta $t{=}0{,}9$ |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 50,00 | 50,00 | 50,00 | 50,00 |
| 50 | 25,00 | **50,00** | 29,73 | 25,17 |
| 20 | 10,00 | **20,00** | 14,95 | 10,16 |
| 5 | 2,50 | **5,00** | 5,00 | 2,58 |
| 1 | 0,50 | **1,00** | 1,00 | 0,52 |

Três coisas para ler nessa tabela:

1. **O indivíduo mais distante não muda de comportamento** (a linha $d = 100$ dá 50,00 em todas as
   colunas): a lei nunca acelera quem está na borda — ela apenas iguala o passo dos demais ao dele.
2. **Quem está no meio ou perto passa a atravessar em vez de rastejar**: o indivíduo a $d = 20$ andava 10 e
   agora anda 20 — chega ao atrator e o ultrapassa via crossover e difusão, em vez de ficar orbitando. É
   isso que impede a nuvem de virar um ponto.
3. **A coluna $t{=}0{,}9$ é praticamente a coluna da L-SHADE** (25,17 contra 25,00; 10,16 contra 10,00): no
   fim do orçamento a proposta *é* a L-SHADE.

#### 3.1.6 Por que isso é melhor — o argumento

A DE contrai de forma **auto-semelhante**: como cada um anda uma fração fixa da própria distância, a razão
$d_i / d_{\max}$ é preservada em média e a nuvem inteira encolhe mantendo a forma. Ninguém nunca atravessa
ninguém, e os indivíduos do miolo dão passos que tendem a zero — eles param de contribuir muito antes de a
busca acabar.

Com o passo de comprimento uniforme, o miolo da nuvem é continuamente **revolvido**: um indivíduo a $d = 5$
dá um passo de 5, isto é, atravessa o incumbente em vez de orbitá-lo. Quem está na borda deixa de ser sugado
em um lance.

E isso é medível. A tabela abaixo é a média de **6 corridas** da F9 em $D{=}30$, medindo, por geração, o
deslocamento que os indivíduos **de fato realizaram** (só os ensaios aceitos pela seleção), em frações do
raio da nuvem:

| $t$ | fator médio<br>L-SHADE | fator médio<br>**proposta** | deslocamento/raio<br>L-SHADE | deslocamento/raio<br>**proposta** | % que se moveu<br>L-SHADE | % que se moveu<br>**proposta** |
| --- | --- | --- | --- | --- | --- | --- |
| $t = 0{,}05$ | 0,722 | **0,884** | 0,0487 | 0,0503 | 10,4% | 10,3% |
| $t = 0{,}20$ | 0,543 | **0,658** | 0,0046 | **0,0085** | 2,2% | **3,8%** |
| $t = 0{,}40$ | 0,508 | **0,669** | 0,0016 | **0,0019** | 5,9% | **6,6%** |
| $t = 0{,}60$ | 0,680 | 0,689 | 0,0007 | **0,0011** | 4,3% | **6,2%** |
| $t = 0{,}80$ | 0,409 | 0,456 | 0,0007 | 0,0004 | 5,6% | 4,2% |
| $t = 0{,}95$ | 0,280 | 0,352 | 0,0003 | 0,0002 | 4,3% | 6,5% |

O que a medição mostra — e o que ela **não** mostra:

- **O fator médio aplicado sobe** cerca de 20 a 30% no primeiro terço (0,884 contra 0,722) e converge para o
  da L-SHADE no fim, exatamente como a rampa manda.
- **O miolo da população se mexe mais.** No meio do ciclo, o deslocamento aceito por geração chega a ser
  **85% maior** ($t = 0{,}20$: 0,0085 contra 0,0046 do raio) e a **fração da população que consegue se mover
  quase dobra** (3,8% contra 2,2%). A nuvem é revolvida mais, geração a geração.
- **A nuvem não termina mais larga.** Medido em 10 corridas, o raio final é $107 \pm 24$ com a proposta e
  $115 \pm 16$ com a original — estatisticamente iguais. O ganho **não** vem de "preservar diversidade" no
  sentido de terminar mais espalhado; vem de **misturar mais** enquanto a busca acontece. Vale registrar
  isso porque a hipótese contrária é a explicação intuitiva, e ela não se sustenta na medição.

Nada disso é reparametrizar $F$: o $F_i$ continua saindo da mesma Cauchy, com a mesma memória de sucesso,
com os mesmos valores. O que muda é **como aquele mesmo $F_i$ é distribuído geometricamente entre os
indivíduos**.

#### 3.1.7 Os dois detalhes que salvam a lei — e o que acontece sem eles

Estas duas peças não são enfeite: sem qualquer uma delas a proposta é **pior** que a L-SHADE.

| peça | o que é | o que acontece se tirar | medido em |
| --- | --- | --- | --- |
| **trava em 1** | $\min(1, \cdot)$ — é o próprio teto de $F$ do SHADE, aplicado depois da graduação | quem está quase em cima do incumbente recebe $d_{\max}/d_i \ggg 1$ e é arremessado para longe: **F2 $D{=}30$ cai de 20/20 para 1/20 acertos** | §5.3 |
| **rampa $(1-t)^2$** | o expoente vai de 1 a 0 conforme o orçamento é gasto | a bent cigar precisa do último terço do ciclo para refinar até $10^{-8}$; sem a rampa **F2 $D{=}50$ cai de 20/20 para 1/20** | §5.3 |

**Nenhuma constante livre nova.** A referência ($d_{\max}$) é a própria geometria da nuvem; a rampa usa a
razão de orçamento $t$ que o LPSR **já calcula**; a trava é a regra de $F$ que o SHADE **já tinha**. O único
grau de liberdade é o expoente 2 da rampa, escolhido por medição em §5.3 (1, 2 e 3 foram testados).

### 3.2 Parte 2 — deflação da bacia exaurida

#### 3.2.1 O problema, em uma frase

Quando a nuvem morre em cima de uma bacia errada, **religar não adianta**, porque a busca nova desce para a
mesma bacia (§2.3). É preciso proibi-la.

#### 3.2.2 O disparo: as duas condições, e só juntas

$$\underbrace{\frac{1}{D}\sum_{j} \frac{\sigma_j(X)}{u_j - l_j} < \tau}_{\text{(A) a nuvem não distingue mais nada}}
\quad\wedge\quad
\underbrace{\text{FES} - \text{FES}_{\text{último ganho}} \geq h \cdot \text{MaxFES}}_{\text{(B) e nada melhora}}$$

com $\tau = 10^{-5}$ (fração da aresta da caixa) e $h = 0{,}05$ (fração do orçamento).

Cada metade **sozinha é destrutiva**, e isso foi medido:

| gatilho | o que acontece | resultado |
| --- | --- | --- |
| só (A), diversidade | religa a F2, a F6 e a F7 enquanto elas ainda estão refinando rumo ao alvo | **F2 $D{=}30$: 30/30 → 0/30 acertos** |
| só (B), estagnação | dispara 12 a 16 vezes por corrida na F9 e na F13, que têm nuvem larga e ganhos raros mas reais | **F13: 0,21 → 0,38** |
| **(A) ∧ (B)** | dispara 1 a 2 vezes por corrida, só na F4 e na F6 | ver §5.1 |

#### 3.2.3 A ação e a deflação

Ao disparar: o ponto de colapso $c$ entra na lista $T$ de bacias proibidas; a população é ressorteada
uniformemente; arquivo e memórias $M_F, M_{CR}$ voltam ao estado inicial; o LPSR continua na rampa do
orçamento global.

A aptidão que a **seleção** enxerga passa a ser

$$\tilde{f}(x) = \begin{cases}
-\infty & \text{se } \min_{c \in T} \lVert x - c\rVert < \rho \\[2pt]
f(x) & \text{caso contrário}
\end{cases}
\qquad \rho = \tfrac{1}{2}\,\overline{(u - l)}$$

Três propriedades desse desenho:

1. **Nenhuma avaliação é desperdiçada.** O valor verdadeiro $f(x)$ continua contando para o incumbente; a
   região proibida muda quem é *selecionado*, não quem é *avaliado*.
2. **A proibição vale durante todo o estágio, não só na hora de sortear a população nova.** É isso que faz
   diferença: a §2.3 mostra que a população *volta* para a bacia se puder. Proibir só na inicialização não
   resolveria.
3. **A lista é esvaziada assim que alguém bate o incumbente.** A partir daí a busca já está numa bacia
   melhor e a proibição só atrapalharia. O mecanismo se desliga sozinho.

O raio $\rho$ = meia aresta da caixa é **a única constante arbitrada do trabalho**, e foi escolhida por
medição (§5.3): abaixo de 60 a bacia não é vencida; acima de 200 a proibição começa a cobrir o próprio
global.

### 3.3 O que mudou no código, arquivo por arquivo

| arquivo | mudança |
| --- | --- |
| `Nature/Models/AdaptiveDE/index.py` | `VARIANTS` ganha `'graded'`; **`pull()`** = parte 1; **`forbid()`, `exhausted()`, `revive()`** = parte 2; **`repair()`** substitui o `np.clip` (§2.5); `update()` passa a rastrear o incumbente separado da população, porque com a deflação a população final pode não conter o melhor de todos |
| `Nature/index.py` | `NatureSelector.MODELS` ganha `'graded_de'` = `AdaptiveDE` com `variant='graded'` fixado |
| `Nature/Processing/Plotter/index.py` | **`curves()`** — as curvas de convergência dos algoritmos no mesmo eixo, exigidas pelo enunciado |
| `Trabalho/Analysis.ipynb` | as doze campanhas ganham o sexto algoritmo, a coluna `p` (Mann-Whitney contra a proposta) e o gráfico de convergência comparativo |

Nenhum código de terceiros foi usado como base. As cinco implementações do pacote `Nature` são próprias; as
fórmulas seguem os artigos citados na §9, e o `deap` é usado apenas no AG.

---

## 4. ESTADO DA ARTE E INEDITISMO

A literatura de variantes de DE é enorme — "mais uma variante de DE" é praticamente um gênero. Por isso a
posição do trabalho é dada trabalho a trabalho, com o que cada um faz, onde testa, que métricas obteve e
onde exatamente ele difere. Os dois mecanismos são analisados separadamente porque a vizinhança de cada um
é diferente.

### 4.1 Vizinhança da Parte 1 (fator de escala por indivíduo governado por distância)

#### A. jSO — o parente mais próximo na *forma* da equação

- **Referência**: J. Brest, M. S. Maučec, B. Bošković, *Single objective real-parameter optimization:
  Algorithm jSO*, IEEE CEC 2017. DOI: [10.1109/CEC.2017.7969456](https://doi.org/10.1109/CEC.2017.7969456)
  (texto fechado; o mecanismo está reproduzido em fontes abertas como o postprint do DISH-XX abaixo).
- **O que faz**: introduz a mutação `DE/current-to-pbest-w/1`, em que o termo de atração recebe um peso
  $F_w$ **separado** do $F$ da difusão: $F_w = 0{,}7F$ enquanto $\text{FES} < 0{,}2\,\text{MaxFES}$;
  $F_w = 0{,}8F$ até $0{,}4\,\text{MaxFES}$; e $F_w = 1{,}2F$ no resto.
- **Onde testa**: CEC 2017, $D \in \{10, 30, 50, 100\}$.
- **Métrica obtida**: 2º lugar na competição CEC 2017 de otimização com restrições de caixa.
- **Por que a proposta não é a mesma coisa**: o jSO **já estabelece que pesar a atração separadamente é uma
  boa ideia** — e é por isso que ele é o parente mais importante. Mas o peso dele é uma **função escada do
  tempo, idêntica para todos os indivíduos**: em qualquer instante, os 540 indivíduos usam o mesmo
  multiplicador. Aqui o peso é **função da posição de cada indivíduo** ($d_{\max}/d_i$), e o tempo entra
  apenas como o expoente que desliga a graduação. Em jSO dois indivíduos, um na borda e outro em cima do
  incumbente, dão passos proporcionais às suas distâncias; aqui eles dão o mesmo passo. É uma diferença de
  natureza, não de valor de constante.

#### B. DISH / Db_SHADE / DbL-SHADE — distância euclidiana, mas em outro lugar

- **Referência**: A. Viktorin, R. Senkerik, M. Pluhacek, T. Kadavy, A. Zamuda, *Distance based parameter
  adaptation for Success-History based Differential Evolution*, Swarm and Evolutionary Computation 50
  (2019) 100462. DOI: [10.1016/j.swevo.2018.10.013](https://doi.org/10.1016/j.swevo.2018.10.013).
  Versão anterior aberta: *L-SHADE Algorithm with Distance Based Parameter Adaptation*,
  DOI: [10.1007/978-3-319-69814-4_7](https://doi.org/10.1007/978-3-319-69814-4_7).
  **PDF aberto que reproduz as equações**:
  [postprint do DISH-XX, UTB](https://publikace.k.utb.cz/bitstream/handle/10563/1009949/Postprint_1009949.pdf?sequence=3).
- **O que faz**: troca o peso da média de Lehmer que atualiza $M_F$ e $M_{CR}$. Em vez de ponderar cada
  sucesso pelo ganho de aptidão, pondera pela **distância euclidiana que o ensaio percorreu do pai**:
  $$w_n = \frac{\lVert u_{n,G} - x_{n,G}\rVert}{\sum_{m=1}^{|S|}\lVert u_{m,G} - x_{m,G}\rVert}$$
- **Onde testa**: CEC 2015 e CEC 2017, em $D \in \{10, 30, 50, 100\}$, aplicado a SHADE, L-SHADE e jSO.
- **Métrica obtida**: as versões com distância (Db_SHADE, DbL_SHADE, DISH) obtêm resultados
  significativamente melhores que as canônicas em 30, 50 e 100 dimensões.
- **Por que a proposta não é a mesma coisa**: **a distância entra em quem é lembrado; aqui ela entra em quem
  dá o passo.** No DISH, o $F$ que um indivíduo aplica continua sendo sorteado da memória sem nenhuma
  relação com a posição dele — dois indivíduos em pontos opostos da nuvem recebem $F$ da mesma
  distribuição. Aqui a atualização da memória é **exatamente a original** (ponderada pelo ganho de aptidão)
  e o que muda é o fator aplicado. Os dois mecanismos são **ortogonais e poderiam ser combinados** — o que
  reforça que não são o mesmo. Motivação declarada, aliás, é a mesma: convergência prematura em alta
  dimensão.

#### C. Vizinhanças espaciais para adaptação (Ghosh et al., 2022)

- **Referência**: A. Ghosh, S. Das, A. K. Das, R. Senkerik, A. Viktorin, I. Zelinka, A. D. Masegosa, *Using
  spatial neighborhoods for parameter adaptation: An improved success history based differential evolution*,
  Swarm and Evolutionary Computation 71 (2022) 101057.
  DOI: [10.1016/j.swevo.2022.101057](https://doi.org/10.1016/j.swevo.2022.101057) ·
  **[PDF aberto (postprint UTB)](https://publikace.k.utb.cz/bitstream/10563/1010978/3/Postprint_1010978.Pdf)**
- **O que faz**: define uma vizinhança por distância espacial em torno do indivíduo corrente e só considera,
  na adaptação de $F$ e $CR$, os sucessos que caem dentro dessa vizinhança — informação local em vez de
  global.
- **Onde testa**: SHADE no CEC 2013, L-SHADE no CEC 2014 e jSO no CEC 2017.
- **Métrica obtida**: melhora significativa sobre o algoritmo original nos três casos.
- **Por que a proposta não é a mesma coisa**: de novo, a distância decide **quais sucessos alimentam a
  adaptação**, não o comprimento do passo. É a mesma família do DISH e a mesma diferença: nenhum indivíduo
  tem o seu $F$ aplicado alterado pela sua própria posição.

#### D. IDE — parâmetros por indivíduo, mas por aptidão

- **Referência**: L. Tang, Y. Dong, J. Liu, *Differential Evolution With an Individual-Dependent Mechanism*,
  IEEE Transactions on Evolutionary Computation 19(4), 2015, 560–574.
  DOI: [10.1109/TEVC.2014.2360890](https://doi.org/10.1109/TEVC.2014.2360890) ·
  **[PDF aberto (repositório Loughborough / figshare)](https://figshare.com/articles/journal_contribution/Differential_evolution_with_an_individual-dependent_mechanism/9501725)**
- **O que faz**: dois mecanismos — IDP, que define $F$ e $CR$ por indivíduo a partir das **diferenças de
  aptidão**, e IDM, que atribui quatro operadores de mutação distintos a indivíduos superiores e inferiores.
- **Onde testa**: as 28 funções do CEC 2013.
- **Métrica obtida**: desempenho competitivo contra as variantes de DE do estado da arte da época.
- **Por que a proposta não é a mesma coisa**: o IDE é o trabalho que estabelece "parâmetro por indivíduo"
  como ideia — e é honesto reconhecer isso. Mas o **controlador é a aptidão**, não a geometria; e o que ele
  atribui são os valores de $F$ e $CR$, não um peso no termo de atração. Duas soluções com a mesma aptidão e
  posições completamente diferentes recebem o mesmo tratamento no IDE e tratamentos opostos aqui.

#### E. div-DE (2025) — o mais parecido em espírito, e na direção **oposta**

- **Referência**: R. Yan, L. Zheng, X. Jin, *Parameter Adaptive Differential Evolution Based on Individual
  Diversity*, Symmetry 17(7), 2025, 1016.
  DOI: [10.3390/sym17071016](https://doi.org/10.3390/sym17071016) (acesso aberto).
- **O que faz**: mede a diversidade de cada indivíduo pela **distância euclidiana dele ao centro da
  população**, gera dois conjuntos simétricos de $F$ e $CR$ e escolhe entre eles pelo ranking de
  diversidade. Indivíduos **próximos** recebem $F$ e $CR$ **menores** (exploração local); indivíduos
  **afastados** recebem $F$ e $CR$ **maiores** (exploração global).
- **Onde testa**: suítes CEC, comparado a cinco variantes de DE do estado da arte.
- **Métrica obtida**: o mecanismo `div` melhora significativamente o DE base e supera as cinco variantes
  comparadas.
- **Por que a proposta não é a mesma coisa** — e este é o ponto mais importante desta seção:
  1. **A referência é outra**: div-DE mede distância ao **centro da população**; aqui é ao **incumbente**
     (o melhor indivíduo), normalizada pelo **raio da nuvem**.
  2. **A direção é oposta — e a direção importa.** div-DE dá $F$ **maior** a quem está longe. Esta proposta
     dá um peso de atração **menor** a quem está longe (para ele não ser engolido) e **maior** a quem está
     perto (para atravessar o incumbente). A direção do div-DE foi implementada na *mesma* lei, com a mesma
     rampa, a mesma trava e as mesmas sementes, e medida:

     | F9 | L-SHADE original | **proposta** ($d_{\max}/d_i$) | direção oposta ($d_i/d_{\max}$) |
     | --- | --- | --- | --- |
     | $D{=}30$, 40 corridas | 16,257 | **11,846** ($p = 0{,}0004$) | 19,827 ($p = 0{,}028$) |
     | $D{=}50$, 30 corridas | 23,297 | **19,984** ($p = 0{,}0014$) | 34,364 ($p < 10^{-4}$) |

     A direção oposta não é apenas "menos boa": ela é **significativamente pior que a própria L-SHADE**, nas
     duas dimensões. O sinal do expoente é a essência do mecanismo, não uma escolha de conveniência.
  3. **O alvo é outro**: div-DE atribui $F$ **e** $CR$ ao indivíduo inteiro; aqui só o termo de atração é
     pesado, e a difusão e o $CR$ ficam intactos.
  4. **Não há teto nem rampa** no div-DE. Aqui, sem a trava em 1 e sem a rampa $(1-t)^2$ a proposta seria
     pior que a original na F2 (§5.3) — as duas peças são parte do mecanismo, não detalhe de implementação.

#### F. ProDE — distância na escolha dos pais

- **Referência**: M. G. Epitropakis, D. K. Tasoulis, N. G. Pavlidis, V. P. Plagianakos, M. N. Vrahatis,
  *Enhancing Differential Evolution Utilizing Proximity-Based Mutation Operators*, IEEE TEVC 15(1), 2011.
  DOI: [10.1109/TEVC.2010.2083670](https://doi.org/10.1109/TEVC.2010.2083670)
- **O que faz**: a probabilidade de escolher $r_1, r_2, r_3$ passa a ser **inversamente proporcional à
  distância** ao indivíduo corrente, favorecendo pais próximos.
- **Onde testa**: CEC 2005 e funções clássicas, sobre várias estratégias de DE.
- **Métrica obtida**: melhora consistente sobre os operadores de mutação canônicos.
- **Por que a proposta não é a mesma coisa**: a distância muda **quem entra na equação**; aqui ela muda
  **por quanto o termo é multiplicado**. São camadas diferentes do mesmo operador e poderiam coexistir.

### 4.2 Vizinhança da Parte 2 (proibir a bacia exaurida)

#### G. ARRDE (2025/26) — o vizinho mais próximo de todo o trabalho

- **Referência**: K. F. Muzakka, A. H. Shali, H. Suhendar, S. Möller, M. Finsterbusch, *Robust Differential
  Evolution via Nonlinear Population Size Reduction and Adaptive Restart: The ARRDE Algorithm*, arXiv, 2025.
  **[PDF aberto](https://arxiv.org/pdf/2511.18429)** · [página](https://arxiv.org/abs/2511.18429)
- **O que faz**: sobre o jSO, junta (i) religamento/refino adaptativo disparado por estagnação, (ii)
  redução não-linear de população dependente da dimensão e (iii) inicialização ciente do orçamento. A
  estagnação é medida por $s = \text{std}(f(P))/\text{mean}(f(P))$, e o religamento é escolhido quando não
  houve melhora e o número de religamentos consecutivos não passou de $N_{\text{rest,max}} = 1 + 4t$.
  Inclui **"local exclusion during restart"**: para cada dimensão $d$, intervalos de exclusão
  $[\ell_d, u_d] = [\max(\mu_d - \sigma_d, L_d),\, \min(\mu_d + \sigma_d, U_d)]$ são construídos a partir das
  populações arquivadas e **os indivíduos novos são reamostrados fora desses intervalos**.
- **Onde testa**: cinco suítes — CEC 2011, 2017, 2019, 2020 e 2022 — com dimensões e orçamentos variados.
- **Métrica obtida**: perfil agregado entre os mais estáveis das cinco suítes; a ablação mostra que o
  religamento/refino é o componente que mais rende, sobretudo em baixa dimensão com orçamento grande, e que
  ele *piora* um pouco o CEC2017 quando usado sem a redução não-linear.
- **Por que a proposta não é a mesma coisa** — três diferenças, e as três são medíveis:
  1. **A geometria da região proibida.** ARRDE exclui uma **caixa alinhada aos eixos** (produto cartesiano
     de intervalos por coordenada). Aqui a região é uma **bola isotrópica no espaço inteiro**. Nas seis
     funções deste enunciado, **todas rotacionadas**, isso é decisivo: uma exclusão alinhada aos eixos não é
     invariante à rotação, e a interseção dela com a bacia real pode ser arbitrariamente ruim.
  2. **O tamanho.** O intervalo do ARRDE é $\pm 1$ desvio-padrão da população arquivada — ou seja, **do
     tamanho da nuvem colapsada**, que na F4 mede $\sigma \approx 5\times10^{-8}$ (§2.1). A tabela de
     sensibilidade da §5.3 mostra que raios abaixo de 60 **não vencem a bacia**: uma exclusão de $\pm\sigma$
     seria, aqui, indistinguível de não ter exclusão nenhuma. O raio desta proposta é macroscópico (meia
     aresta da caixa).
  3. **Onde a exclusão age.** No ARRDE ela age **na reamostragem** da população nova. Aqui ela age **na
     seleção, durante todo o estágio**. Essa é a diferença que a §2.3 mede: religar *fora* da bacia não
     resolve, porque a população **desce de volta** para dentro dela — foi exatamente isso que aconteceu em
     30 de 30 corridas no braço "religa sem tabu" (§5.2). Proibir só a largada não impede a chegada.

  Vale registrar que o ARRDE é contemporâneo (submetido em novembro de 2025) e que a convergência
  independente para a ideia de "excluir região já visitada em DE" reforça que o problema é real; o que
  muda é a formulação.

#### H. RR-CMA-ES — repulsão feita direito, mas em CMA-ES

- **Referência**: J. de Nobel, D. Vermetten, A. V. Kononova, O. M. Shir, T. Bäck, *Avoiding Redundant
  Restarts in Multimodal Global Optimization*, arXiv:2405.01226, 2024.
  **[PDF aberto](https://arxiv.org/pdf/2405.01226)** · [página](https://arxiv.org/abs/2405.01226)
- **O que faz**: mede quanto do orçamento é gasto por reinícios que caem numa bacia já visitada (o
  *redundancy potential*, análogo ao Problema do Colecionador de Cupons) e propõe o RR-CMA-ES, que mantém
  um arquivo $T$ de pontos tabu. Cada ponto amostrado é **rejeitado antes de ser avaliado** se
  $$d_m(x, x_T, C^{-1})/\sigma \;<\; \gamma^{\,n_{\text{rej}}}\,\delta(T)$$
  com $d_m$ a **distância de Mahalanobis** escalada pela matriz de covariância $C$ e pelo passo $\sigma$ do
  próprio CMA-ES. Usa ainda a heurística **Hill-Valley** (até 10 avaliações extras por teste) para decidir
  se dois pontos pertencem à mesma bacia.
- **Onde testa**: as 24 funções do BBOB/COCO, em todas as dimensões e instâncias.
- **Métrica obtida**: reduz a fração de orçamento gasta com reinícios redundantes; os autores registram que
  o ganho **exige calibração cuidadosa**, sob pena de deteriorar o desempenho em alguns tipos de relevo.
- **Por que a proposta não é a mesma coisa**:
  1. **Base diferente e escala indisponível.** O RR-CMA-ES define a região com $C$ e $\sigma$ — quantidades
     que o CMA-ES mantém por construção e que **a DE simplesmente não tem**. Transportar a ideia para DE
     exige inventar a escala, que é o problema que a §5.3 resolve por medição.
  2. **Rejeição antes vs. depois da avaliação.** Lá o ponto é rejeitado **sem ser avaliado**; aqui ele é
     avaliado, a avaliação **conta para o incumbente**, e a rejeição acontece na seleção. Nenhuma avaliação
     é jogada fora — e o incumbente pode, inclusive, ser encontrado dentro da região proibida.
  3. **Sem custo de diagnóstico.** O Hill-Valley gasta até 10 avaliações por teste de bacia. O disparo aqui
     usa duas quantidades que o algoritmo **já calcula de graça**: o desvio da população e o contador de
     avaliações desde o último ganho.
  4. **Autodesligamento.** Aqui a lista é esvaziada quando o incumbente é batido; o RR-CMA-ES encolhe a
     região por $\gamma^{n_{\text{rej}}}$ para não travar, o que é um remédio para o mesmo problema, com
     outra formulação e mais uma constante.

#### I. Religamento em DE sem exclusão

- **ADE-DMRM**: *An adaptative differential evolution with enhanced diversity and restart mechanism*, Expert
  Systems with Applications, 2024. DOI:
  [10.1016/j.eswa.2024.123634](https://doi.org/10.1016/j.eswa.2024.123634).
- **R-SHADE / RL-SHADE**: variantes com religamento descritas na literatura de SHADE.
- **O que fazem**: reinicializam a população quando o melhor não melhora por um número de gerações, para
  recuperar diversidade.
- **Por que a proposta não é a mesma coisa**: são exatamente o braço de ablação **"religa sem tabu"**, que
  neste relevo devolve $76{,}5754258$ em **30 de 30 corridas** (§5.2) — indistinguível de não fazer nada. O
  religamento não é a contribuição; a proibição é.

#### J. Niching com repulsão (origem da ideia)

- **Referência**: A. Ahrari, K. Deb, M. Preuss, *Multimodal optimization by covariance matrix
  self-adaptation evolution strategy with repelling subpopulations*, Evolutionary Computation 25(3), 2017.
  DOI: [10.1162/evco_a_00182](https://doi.org/10.1162/evco_a_00182)
- **O que faz**: subpopulações que se repelem para localizar **vários** ótimos (RS-CMSA-ES).
- **Por que a proposta não é a mesma coisa**: o objetivo do *niching* é **coletar** os ótimos; aqui o
  objetivo é o oposto — descartar os ótimos já coletados para achar **um** melhor, dentro do orçamento fixo
  de uma competição mono-objetivo.

### 4.3 O que é inédito, em uma frase cada

1. Usar a razão $d_{\max}/d_i$ — distância de cada indivíduo ao incumbente sobre o raio da nuvem — como
   **peso do termo de atração**, de modo que o comprimento do passo fique uniforme, com a saturação em 1 do
   SHADE e a rampa $(1-t)^2$. Não foi encontrada variante de SHADE/L-SHADE/jSO que pese a atração pela
   geometria do indivíduo; as que usam distância a colocam na **memória** (DISH, Ghosh) ou na **escolha dos
   pais** (ProDE), e a que a coloca no parâmetro usa a distância **ao centro** e na **direção oposta**
   (div-DE).
2. Condicionar o religamento à **conjunção** nuvem-morta ∧ sem-progresso, e proibir a bacia exaurida **na
   seleção durante todo o estágio**, com bola isotrópica de raio macroscópico e autodesligamento ao bater o
   incumbente. O trabalho mais próximo (ARRDE) exclui **caixas alinhadas aos eixos, do tamanho da nuvem,
   apenas na reamostragem**; o outro (RR-CMA-ES) faz repulsão bem-feita, mas em CMA-ES e com custo extra de
   avaliações.

### 4.4 Sobre comparar **números** com a literatura

O enunciado pede comparação com algoritmos recentes, e ela é feita aqui no plano do mecanismo, não no dos
valores. O motivo é honesto e está na §7: o deslocamento $o$ e a rotação $M$ deste experimento são sorteados
com semente fixa por função, pela construção da seção 1.1 do relatório do CEC'14, em vez de lidos dos
arquivos `shift_data_*.txt` e `M_*_D*.txt` distribuídos com a competição. O relevo é estatisticamente
equivalente e o protocolo é idêntico, mas **os erros absolutos não se comparam, número a número**, com as
tabelas publicadas de jSO, LSHADE-cnEpSin, LSHADE-SPACMA ou EBOwithCMAR — esta instância da F4, por exemplo,
tem uma bacia dominante em $76{,}575$ que a instância oficial não tem necessariamente. Uma comparação
numérica com a literatura exige trocar as doze células pelos arquivos oficiais (disponíveis no pacote
`opfunu`, em `opfunu/cec_based/data_2014`). O mecanismo, as equações, a ablação e a comparação contra a
L-SHADE original valem exatamente como estão.

---

## 5. RESULTADOS

Protocolo do enunciado, 30 execuções independentes por campanha, sementes fixas (7000–7029) para que a
tabela seja reproduzível. `p` é o Mann-Whitney bilateral entre as 30 amostras de erro dos dois braços. Os
dois braços são **o mesmo código**, diferindo apenas por `variant`.

### 5.1 As doze campanhas

Erro $= f - F^*$, com erro $< 10^{-8}$ contado como zero (seção 2.1 do relatório do CEC'14). `ok` é o
número de corridas que atingiram o alvo.

| campanha | L-SHADE mediana | média | ok | **proposta** mediana | média | ok | $p$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 $D{=}30$ | 0 | 0 | 30/30 | 0 | 0 | 30/30 | — |
| **F4 $D{=}30$** | 76,5754 | 76,575 | 0/30 | **4,569** | **30,99** | 0/30 | **$<10^{-4}$** |
| F6 $D{=}30$ | 0 | 0,3861 | 24/30 | 0 | 0,2761 | 22/30 | 0,74 |
| F7 $D{=}30$ | 0 | 0 | 30/30 | 0 | 0 | 30/30 | — |
| **F9 $D{=}30$** | 15,863 | 16,898 | 0/30 | **12,453** | **13,543** | 0/30 | **0,0042** |
| F13 $D{=}30$ | 0,20010 | 0,18927 | 0/30 | 0,17961 | 0,18232 | 0/30 | 0,21 |
| F2 $D{=}50$ | 0 | 0 | 30/30 | 0 | 0 | 30/30 | — |
| F4 $D{=}50$ | 71,069 | 67,447 | 0/30 | 72,838 | 75,159 | 0/30 | 0,085 |
| F6 $D{=}50$ | 2,8902 | 3,0245 | 3/30 | 2,7599 | 2,9452 | 1/30 | 0,93 |
| F7 $D{=}50$ | 0 | 0 | 30/30 | 0 | 0 | 30/30 | — |
| **F9 $D{=}50$** | 23,499 | 23,955 | 0/30 | **17,845** | **18,437** | 0/30 | **$<10^{-4}$** |
| F13 $D{=}50$ | 0,25565 | 0,25051 | 0/30 | 0,23412 | 0,24090 | 0/30 | 0,093 |

Leitura, campanha a campanha:

- **Três vitórias estatisticamente significativas.** A F4 em $D{=}30$ é a mais expressiva: a mediana cai de
  $76{,}5754$ para $4{,}569$ — **17 vezes menor** — e 13 das 30 corridas passam a fugir da bacia
  ($\text{erro} < 1$), contra **0 de 30** da original. É exatamente a campanha em que a L-SHADE não chegava
  ao objetivo, e em que ela falhava de forma idêntica nas 30 corridas. A F9 (rastrigin rotacionada) melhora
  nas duas dimensões — 21% em $D{=}30$ e **24% em $D{=}50$** — e é a função mais multimodal da suíte.
- **Nenhum dano onde a original já resolvia.** F2 e F7, nas duas dimensões, dão 30/30 nos dois braços: todas
  as corridas dos dois algoritmos atingem o alvo, então as amostras de erro são iguais (todas nulas). Isso
  não é sorte — é o que a trava em 1, a rampa $(1-t)^2$ e a conjunção do disparo foram construídas para
  garantir, cada uma delas adicionada **depois** de medir o estrago que a sua ausência causava.
- **Duas melhoras não significativas na direção certa**: F13 nas duas dimensões ($p = 0{,}21$ e $0{,}093$).
- **Uma piora, não significativa**: F4 em $D{=}50$ ($p = 0{,}085$). A ablação mostra que a culpa é da parte
  1, e a §7 explica por quê.

### 5.2 Ablação — qual parte responde por quê

Mesmas 30 sementes; cada braço é o próprio código do pacote com uma das partes desligada. `<1` conta as
corridas que saíram da bacia dominante.

**F4 $D{=}30$** — a campanha do colapso:

| braço | mediana | média | `<1` | $p$ vs. original |
| --- | --- | --- | --- | --- |
| L-SHADE fiel | 76,575 | 76,575 | 0/30 | — |
| só a parte 1 (lei) | 76,575 | 76,575 | 0/30 | 0,33 |
| só a parte 2 (deflação) | 35,663 | 35,314 | 13/30 | $<10^{-4}$ |
| religa **sem** proibir a bacia | 76,575 | 75,779 | 0/30 | 0,013 |
| **proposta (as duas)** | **4,569** | **30,991** | 13/30 | $<10^{-4}$ |

1. **É o tabu, não o religamento.** Religar na estagnação sem proibir a bacia devolve $76{,}5754258$ em 30
   de 30 corridas. O mecanismo inteiro está na proibição.
2. **A lei sozinha não escapa** — e não deveria: ela reescala *dentro* da nuvem, e quando a nuvem colapsa
   todas as distâncias colapsam junto. Nenhuma lei invariante de escala pode ressuscitar a busca.
3. **As duas juntas são melhores que a soma.** O número de fugas é o mesmo (13/30), mas a mediana cai de
   $35{,}7$ para $4{,}6$: a lei não abre a porta — ela faz a corrida que fugiu **descer mais fundo** na
   bacia nova, porque cada estágio depois do religamento tem menos orçamento e precisa convergir mais rápido.

**F4 $D{=}50$** — de onde vem a única piora:

| braço | mediana | média | `<1` | $p$ |
| --- | --- | --- | --- | --- |
| L-SHADE fiel | 71,069 | 67,447 | 5/30 | — |
| só a parte 1 (lei) | 72,838 | 78,627 | 2/30 | 0,080 |
| só a parte 2 (deflação) | 71,069 | 64,769 | 5/30 | 0,99 |
| proposta | 72,838 | 75,159 | 2/30 | 0,085 |

A deflação é **exatamente inerte** aqui (mesmos 5/30, $p = 0{,}99$): em $D{=}50$ a nuvem só morre no fim do
orçamento e o disparo quase nunca acontece. A perda é da lei.

**F9 $D{=}30$** — a campanha da lei, e o espelho exato da F4:

| braço | mediana | média | $p$ |
| --- | --- | --- | --- |
| L-SHADE fiel | 15,863 | 16,898 | — |
| **só a parte 1 (lei)** | **12,453** | **13,543** | **0,0042** |
| só a parte 2 (deflação) | 15,863 | 16,898 | 1,00 |
| **proposta** | **12,453** | **13,543** | **0,0042** |

Aqui a deflação devolve a **amostra idêntica** à da original — as 30 corridas, valor por valor. Não é
aproximação: na F9 a população nunca fica numericamente morta, o disparo nunca acontece e a parte 2 não
executa uma única linha.

**F6** — empate nas duas dimensões: todos os braços entre $p = 0{,}45$ e $p = 0{,}93$. Em $D{=}30$ a
deflação melhora as corridas presas (`<1` sai de 24/30 para 26/30) mas o número de acertos exatos oscila
dentro do ruído. Nenhuma conclusão pode ser tirada desta campanha.

**A inércia, medida.** Em quatro das doze campanhas a parte 2 devolve amostra **bit-idêntica** à da L-SHADE
original — F9 $D{=}30$, F13 $D{=}50$ e, no braço completo, F2 e F7 nas duas dimensões. Não é "diferença
pequena": o disparo conjunto simplesmente nunca acontece e o algoritmo *é* a L-SHADE.

**Resumo.** A parte 2 responde pela F4 em $D{=}30$ — e só ela. A parte 1 responde pela F9 nas duas dimensões
e pela F13 — e só ela. Elas se somam num único lugar, a F4 em $D{=}30$.

### 5.3 Sensibilidade — cada escolha foi medida, não arbitrada

**A rampa da graduação** (40 corridas em F9, 20 em F2, sementes 9000+; parte 2 desligada):

| expoente | F2 $D{=}50$ (acertos) | F9 $D{=}30$ | F9 $D{=}50$ | F13 $D{=}30$ | F6 $D{=}30$ (acertos) |
| --- | --- | --- | --- | --- | --- |
| L-SHADE fiel | 20/20 | 17,8 | 23,7 | 0,199 | 23/30 |
| sem rampa | 1/20 | 13,3 | 17,2 | 0,175 | 15/30 |
| $(1-t)^1$ | 15/20 | 13,2 | 17,5 | 0,174 | 16/30 |
| **$(1-t)^2$** | **20/20** | **12,4** | 18,3 | 0,178 | 19/30 |
| $(1-t)^3$ | 20/20 | 14,4 | 19,2 | 0,187 | 19/30 |

Sem rampa a F9 fica ótima e a F2 morre; com rampa cúbica a F2 se salva e a F9 perde metade do ganho. O
quadrado é o único ponto que preserva as duas.

**A referência de distância** (F9 $D{=}30$, dois blocos de sementes independentes):

| referência | mediana | $p$ (bloco A) | $p$ (bloco B) |
| --- | --- | --- | --- |
| mediana das distâncias | 13,7 / 16,5 | 0,031 | 0,37 |
| **máximo (raio da nuvem)** | **13,1** | $<10^{-4}$ | $<10^{-4}$ |

A versão com a mediana **não replica**: significativa num bloco de sementes e nula noutro. Foi o que
motivou trocar a referência pelo raio da nuvem, e é a razão de todo número deste documento vir com teste de
hipótese — numa DE, diferenças de 3 ou 4 acertos em 30 são ruído.

**A direção da lei** (F9, sementes 9000+, mesma rampa e mesma trava nos dois sentidos):

| | L-SHADE | proposta ($d_{\max}/d_i$) | direção oposta ($d_i/d_{\max}$) |
| --- | --- | --- | --- |
| $D{=}30$ (40 corridas) | 16,257 | **11,846** ($p = 0{,}0004$) | 19,827 ($p = 0{,}028$, **pior**) |
| $D{=}50$ (30 corridas) | 23,297 | **19,984** ($p = 0{,}0014$) | 34,364 ($p < 10^{-4}$, **pior**) |

Inverter o sinal do expoente não devolve a L-SHADE: devolve algo pior que ela. É a evidência mais direta de
que o ganho vem do mecanismo geométrico e não de "mexer no $F$".

**O raio da bacia proibida** (F4 $D{=}30$, 15 corridas):

| $\rho$ | 30 | 60 | **100 = ½ aresta** | 150 | 200 |
| --- | --- | --- | --- | --- | --- |
| mediana | 76,58 | 76,58 | **0,27** | 67,80 | 4,01 |

Abaixo de 60 a proibição é pequena demais e a nuvem re-converge para a borda da bola; a partir de 150 ela
começa a cobrir região útil. Meia aresta é o centro do intervalo que funciona.

---

## 6. MAPA DOS REQUISITOS DO ENUNCIADO

Cada linha é uma exigência do arquivo `Trabalho/files/Trabalho final - 2026.2.pdf`, com onde ela é cumprida
e como verificar.

| # | requisito do enunciado | como foi cumprido | onde conferir |
| --- | --- | --- | --- |
| 1 | **Escolher um método de referência da disciplina** | L-SHADE, a mais forte das cinco implementações do pacote `Nature` e vencedora do CEC'14 | §1; `variant='lshade'` |
| 2 | **Propor modificação inédita** (variar parametrização **não** é modificar) | duas mudanças na mecânica: o termo de atração passa a ter fator de escala por indivíduo governado pela distância (§3.1), e a bacia exaurida passa a ser proibida na seleção (§3.2). Nenhum parâmetro da L-SHADE foi retocado: $p = 0{,}11$, arquivo $2{,}6N$, $H = 6$, $N_{\text{init}} = 18D$, $N_{\min} = 4$ seguem os valores publicados | §3; §4 posiciona contra 10 trabalhos |
| 3 | **Implementar a modificação** | quarta variante da classe `AdaptiveDE`, 4 métodos novos e 1 corrigido | §3.3; `Nature/Models/AdaptiveDE/index.py` |
| 4 | **30 execuções para cada função** | `n_gaussian=30` nas doze células; a tabela da §5.1 também é de 30 corridas | §5.1; células do notebook |
| 5 | **…assim como com a versão original, para comparação** | `adaptive_de` (L-SHADE fiel) roda lado a lado em todas as células, com o mesmo código-base | §5.1; toda célula tem os dois |
| 6 | **Analisar conforme o objetivo da modificação** | o objetivo foi fixado pelo diagnóstico (§2) e cada parte é avaliada contra a falha que ela ataca, com ablação isolando as contribuições | §2, §5.2 |
| 7 | **Estatísticas dos resultados** | mínimo, mediana, média, desvio, pior, acertos e **Mann-Whitney bilateral** (coluna `p`) por campanha; distribuição e Q-Q das 30 corridas do vencedor | §5.1; quadro e figuras de cada célula |
| 8 | **Gráficos de evolução** | curvas de mínimo/máximo por geração do vencedor (`plotMetrics` via `Portrait`), faixa das variáveis e trajetória temporal (`plotVariables`) | figuras de cada célula |
| 9 | **Gráficos de convergência** | erro contra avaliações gastas, **os seis algoritmos no mesmo eixo**, escala log com a linha de tolerância $10^{-8}$ | `Plotter.curves()`, figura de cada célula |
| 10 | **Todos os recursos que sustentem as afirmações** | diagnóstico quantitativo (§2), ablação por mecanismo (§5.2), três tabelas de sensibilidade (§5.3), verificação da contagem de avaliações e da inércia bit-a-bit | §2, §5.2, §5.3 |
| 11 | **Comparar com outros algoritmos recentes da literatura** | dez trabalhos analisados um a um (o que faz, onde testa, métricas, diferença), com link de acesso; e a ressalva honesta sobre comparação numérica | §4 |
| 12 | **MaxFES $= 10000\cdot D$** | `MAXFES = 10000 * D` e `generations = MAXFES // population`; verificado: a corrida gasta 299 703 de 300 000 em $D{=}30$ e nunca ultrapassa | célula 1 e todas as células |
| 13 | **Espaço de busca $[-100,100]^D$** | `{'bounds': (-100.0, 100.0)}` para as $D$ variáveis, em todas as células | todas as células |
| 14 | **Inicialização não-determinística** | `seed=None` no dicionário `problem`: cada uma das 30 corridas sorteia a própria população | todas as células |
| 15 | **Parada: erro $< 10^{-8}$ ou fim do orçamento** | `target = F* + 1e-8` encerra a corrida no lance em que o alvo é atingido; o orçamento encerra o resto | todas as células |
| 16 | **Funções F2, F4, F6, F7, F9, F13** | uma célula por função e dimensão, com a fórmula escrita na própria célula e conferida contra o relatório do CEC'14 | células 3–19 |
| 17 | **$D = 30$ e $D = 50$** | doze campanhas, seis por dimensão | células 3–19 |
| 18 | **Relatório com estrutura de artigo científico** | este documento tem a estrutura (contexto → diagnóstico → método → trabalhos relacionados → resultados → limitações → referências); o texto para o modelo IEEE Access de `Trabalho/files/` sai daqui sem reescrita | este arquivo |
| 19 | **Usar os documentos auxiliares** | as fórmulas, os $F^*$ da tabela I e o protocolo da seção 2.1 vêm de `Definitions of CEC2014 benchmark suite.pdf`; o modelo do artigo é o `Access-Template.pdf` / `access.zip` | §2, §9 e as células |
| 20 | **Citar todas as fontes; explicitar locais de intervenção se usar código de terceiros** | nenhum código de terceiros foi usado como base — as implementações são próprias, com as fórmulas dos artigos citados; os pontos de intervenção estão listados método a método | §3.3, §9 |
| 21 | **Submeter códigos, relatório e apresentação** | códigos e experimento no repositório; relatório neste arquivo; **a apresentação ainda precisa ser montada** | pendente |
| 22 | **Apresentação oral de 15 minutos** | pendente | pendente |

**Aberto:** os itens 21 e 22 (apresentação) e, se a comparação numérica com a literatura for exigida, a
troca do relevo sorteado pelos arquivos oficiais do CEC'14 (§4.4).

---

## 7. LIMITAÇÕES

- **F4 em $D{=}50$ piora** (não significativamente, $p = 0{,}085$), e a culpa é da parte 1. A rosenbrock
  rotacionada é um vale estreito e curvo: o passo de comprimento uniforme, que na rastrigin serve para
  atravessar bacias, aqui joga para fora do vale os indivíduos que já estavam dentro dele. É o preço de uma
  lei puramente geométrica, que não distingue "bacia vizinha" de "parede do vale".
- **A parte 2 quase não age em $D{=}50$.** O disparo exige a nuvem numericamente morta, e em 50 dimensões
  isso só acontece perto do fim do orçamento — não sobra ciclo para o religamento render. O ganho da
  deflação é, na prática, um fenômeno de $D{=}30$.
- **A F6 fica no empate** nas duas dimensões: média um pouco melhor, acertos um pouco piores, $p$ perto de
  1.
- **O raio $\rho$ é a única constante arbitrada**, e é fração da caixa — não escala com $D$. O intervalo
  útil é largo (100–200 numa caixa de aresta 200), mas transportar o mecanismo para outra caixa exige
  refazer a medição da §5.3.
- **Relevo sorteado**, com a consequência descrita na §4.4: comparações entre algoritmos valem
  integralmente; valores absolutos não se comparam com as tabelas publicadas do CEC'14.
- **Uma variante, uma suíte.** A proposta foi avaliada nas seis funções do enunciado, em duas dimensões.
  Nada aqui autoriza extrapolar para as 30 funções da suíte completa nem para outras dimensões.

---

## 8. COMO REPRODUZIR

```python
from Nature.index import NatureSelector

# a proposta
NatureSelector('graded_de',   {**problem, 'population': 18 * D,
                               'generations': MAXFES // (18 * D)}, n_gaussian=30)

# o braço de comparação: a mesma classe, com a L-SHADE fiel
NatureSelector('adaptive_de', {**problem, 'population': 18 * D,
                               'generations': MAXFES // (18 * D),
                               'variant': 'lshade'}, n_gaussian=30)
```

As doze células do `Analysis.ipynb` rodam os dois lado a lado, com os outros quatro algoritmos do pacote, e
imprimem o quadro com a coluna `p`, as barras comparativas, a curva de convergência dos seis no mesmo eixo,
a distribuição e o Q-Q das 30 corridas do vencedor e os cortes do relevo. A campanha inteira leva cerca de
3 horas de CPU.

A tabela da §5.1 foi levantada à parte, com sementes fixas (7000–7029) em vez de `seed=None`, para ser
reproduzível linha a linha; o notebook usa `seed=None`, como o enunciado exige, então os números dele variam
dentro das barras de erro dadas aqui.

Uma consequência lateral da correção da §2.5: o `adaptive_de` do `Analysis.ipynb` **da raiz** também mudou
de comportamento (para melhor). Os resultados gravados naquele notebook são anteriores à correção. A versão
do notebook do trabalho anterior à proposta está preservada em
`Trabalho/files/Analysis-antes-da-proposta.ipynb`.

---

## 9. REFERÊNCIAS

**Método de referência e antecedentes diretos**

1. R. Tanabe, A. Fukunaga. *Improving the search performance of SHADE using linear population size
   reduction*. IEEE CEC 2014. DOI: [10.1109/CEC.2014.6900380](https://doi.org/10.1109/CEC.2014.6900380) ·
   **[PDF](https://ryojitanabe.github.io/pdf/tf-cec2014.pdf)**
2. R. Tanabe, A. Fukunaga. *Success-history based parameter adaptation for differential evolution* /
   *Evaluating the performance of SHADE on CEC 2013 benchmark problems*. IEEE CEC 2013.
   **[PDF](https://ryojitanabe.github.io/pdf/tf-cec2013-compe.pdf)**
3. J. Zhang, A. C. Sanderson. *JADE: Adaptive differential evolution with optional external archive*. IEEE
   TEVC 13(5), 2009. DOI: [10.1109/TEVC.2009.2014613](https://doi.org/10.1109/TEVC.2009.2014613)

**Trabalhos próximos à Parte 1**

4. J. Brest, M. S. Maučec, B. Bošković. *Single objective real-parameter optimization: Algorithm jSO*. IEEE
   CEC 2017. DOI: [10.1109/CEC.2017.7969456](https://doi.org/10.1109/CEC.2017.7969456)
5. A. Viktorin, R. Senkerik, M. Pluhacek, T. Kadavy, A. Zamuda. *Distance based parameter adaptation for
   success-history based differential evolution*. Swarm Evol. Comput. 50 (2019) 100462.
   DOI: [10.1016/j.swevo.2018.10.013](https://doi.org/10.1016/j.swevo.2018.10.013) ·
   **[PDF aberto com as equações (DISH-XX)](https://publikace.k.utb.cz/bitstream/handle/10563/1009949/Postprint_1009949.pdf?sequence=3)**
6. A. Ghosh, S. Das, A. K. Das, R. Senkerik, A. Viktorin, I. Zelinka, A. D. Masegosa. *Using spatial
   neighborhoods for parameter adaptation: An improved success history based differential evolution*.
   Swarm Evol. Comput. 71 (2022) 101057.
   DOI: [10.1016/j.swevo.2022.101057](https://doi.org/10.1016/j.swevo.2022.101057) ·
   **[PDF](https://publikace.k.utb.cz/bitstream/10563/1010978/3/Postprint_1010978.Pdf)**
7. L. Tang, Y. Dong, J. Liu. *Differential evolution with an individual-dependent mechanism*. IEEE TEVC
   19(4), 2015. DOI: [10.1109/TEVC.2014.2360890](https://doi.org/10.1109/TEVC.2014.2360890) ·
   **[PDF](https://figshare.com/articles/journal_contribution/Differential_evolution_with_an_individual-dependent_mechanism/9501725)**
8. R. Yan, L. Zheng, X. Jin. *Parameter Adaptive Differential Evolution Based on Individual Diversity*.
   Symmetry 17(7), 2025, 1016. DOI: [10.3390/sym17071016](https://doi.org/10.3390/sym17071016) (aberto)
9. M. G. Epitropakis, D. K. Tasoulis, N. G. Pavlidis, V. P. Plagianakos, M. N. Vrahatis. *Enhancing
   differential evolution utilizing proximity-based mutation operators*. IEEE TEVC 15(1), 2011.
   DOI: [10.1109/TEVC.2010.2083670](https://doi.org/10.1109/TEVC.2010.2083670)

**Trabalhos próximos à Parte 2**

10. K. F. Muzakka, A. H. Shali, H. Suhendar, S. Möller, M. Finsterbusch. *Robust Differential Evolution via
    Nonlinear Population Size Reduction and Adaptive Restart: The ARRDE Algorithm*. arXiv:2511.18429, 2025.
    **[PDF](https://arxiv.org/pdf/2511.18429)**
11. J. de Nobel, D. Vermetten, A. V. Kononova, O. M. Shir, T. Bäck. *Avoiding redundant restarts in
    multimodal global optimization*. arXiv:2405.01226, 2024. **[PDF](https://arxiv.org/pdf/2405.01226)**
12. *An adaptative differential evolution with enhanced diversity and restart mechanism*. Expert Systems
    with Applications, 2024. DOI: [10.1016/j.eswa.2024.123634](https://doi.org/10.1016/j.eswa.2024.123634)
13. A. Ahrari, K. Deb, M. Preuss. *Multimodal optimization by covariance matrix self-adaptation evolution
    strategy with repelling subpopulations*. Evolutionary Computation 25(3), 2017.
    DOI: [10.1162/evco_a_00182](https://doi.org/10.1162/evco_a_00182)

**Suíte de teste**

14. J. J. Liang, B. Y. Qu, P. N. Suganthan. *Problem definitions and evaluation criteria for the CEC 2014
    special session and competition on single objective real-parameter numerical optimization*. 2013.
    (`Trabalho/files/Definitions of CEC2014 benchmark suite.pdf`)
