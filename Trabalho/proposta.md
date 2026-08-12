# PROPOSTA DE MODIFICAÇÃO — DE COM ESCALA GEOMÉTRICA (DE-g)

Método de referência: **DE** clássica (Storn & Price, 1997), na configuração `current-to-best/1/bin` que o
pacote `Nature` já implementa e que serve de baseline. A modificação proposta troca **uma fórmula** dentro
do operador de mutação; tudo o mais — população, orçamento, crossover, seleção, tratamento de fronteira —
fica idêntico entre as duas versões.

## O QUE MUDA

A DE clássica sorteia um único fator de escala por geração (*dither* de Storn & Price) e aplica o mesmo
valor a todos os indivíduos:

$$f^{(g)} \sim U[F_{min}, F_{max}], \qquad
v_i = x_i + f^{(g)}\,(x_{best} - x_i) + f^{(g)}\,(x_{r_1} - x_{r_2})$$

A proposta é dar a cada indivíduo o seu próprio fator, governado pela **distância geométrica dele ao
incumbente**:

$$d_i = \frac{\lVert x_i - x_{best}\rVert}{\max_j \lVert x_j - x_{best}\rVert} \in [0,1], \qquad
F_i = F_{min} + (F_{max} - F_{min})\,d_i$$

$$v_i = x_i + F_i\,(x_{best} - x_i) + F_i\,(x_{r_1} - x_{r_2})$$

O ponto de intervenção é o método `donor()`: o escalar `f` vira um vetor `F` de comprimento `population`.
Nenhum parâmetro novo — $F_{min}$ e $F_{max}$ já são os extremos do dither que a DE original usa. Nenhuma
avaliação extra, nenhuma memória, nenhum contador: as duas versões gastam exatamente o mesmo MaxFES e podem
rodar com as mesmas sementes, o que torna a comparação **pareada** (Wilcoxon signed-rank, não rank-sum).

## A HIPÓTESE

O `F` único trata a população como homogênea, e ela não é. Quem está perto do melhor deveria refinar; quem
está longe deveria explorar. Hoje os dois recebem o mesmo passo, e o resultado é o **colapso de passo**: o
termo diferencial $(x_{r_1} - x_{r_2})$ encolhe junto com a população, e a partir do momento em que ela se
contrai a DE não consegue mais reexpandir — não existe mecanismo, na formulação clássica, que devolva
amplitude à busca.

Com $F_i \propto d_i$, os indivíduos da borda seguram passo grande enquanto o núcleo refina. A divisão de
trabalho acontece **dentro de uma população só**, sem subpopulações, sem reinício e sem custo.

Evidência preliminar do modo de falha, DE original em $D=30$ com $10^5$ avaliações e 3 sementes:
$F_9$ (rastrigin) estaciona em erro **193**, com a população inteira colapsada numa bacia errada. É o cenário
que a fórmula ataca.

## POR QUE ESSAS SEIS FUNÇÕES

| função | o que ela cobra | o que a fórmula faz |
| --- | --- | --- |
| $F_2$ bent cigar | condicionamento $10^6$, vale estreito | núcleo com $F_i$ pequeno refina na direção folgada sem sair do vale |
| $F_4$ rosenbrock | vale curvo e estreito | passo curto perto do incumbente, que é onde a curva do vale exige precisão |
| $F_6$ weierstrass | platôs, sem gradiente | **a aptidão não separa os indivíduos, mas a geometria separa** |
| $F_7$ griewank | ondulação fina sobre parábola larga | borda com passo grande atravessa a ondulação; núcleo desce a parábola |
| $F_9$ rastrigin | $10^{30}$ mínimos locais, rotacionada | é o caso central: impede o colapso que estaciona a DE em 193 |
| $F_{13}$ happycat | mínimo sobre casca fina, fundo achatado | mesma situação da $F_6$: sem sinal de aptidão, sobra a distância |

O argumento das $F_6$ e $F_{13}$ é o mais forte da proposta e é ele que a separa do trabalho publicado mais
próximo (ver adiante): **em região plana o posto de aptidão não carrega informação, e a distância carrega.**

## AS DUAS VARIANTES — O MIOLO DA ANÁLISE

A escolha do normalizador de $d_i$ produz dois algoritmos de comportamento oposto no fim da corrida, e
comparar os dois **é** o conteúdo científico do trabalho:

**Relativa** — divide pelo $\max_j$ da geração corrente. A fórmula é auto-similar: a população encolhe e a
distribuição de $F_i$ continua a mesma. O mecanismo nunca desliga, o que preserva diversidade até o fim mas
pode impedir o fechamento da convergência.

**Absoluta** — divide pela diagonal da caixa inicial, $\lVert up - low \rVert$. Aí $F_i$ encolhe junto com a
população: recupera-se a convergência final, ao custo de parte do efeito anti-colapso.

$$d_i^{rel} = \frac{\lVert x_i - x_{best}\rVert}{\max_j \lVert x_j - x_{best}\rVert}
\qquad\text{contra}\qquad
d_i^{abs} = \frac{\lVert x_i - x_{best}\rVert}{\lVert up - low\rVert}$$

Hipótese a testar: a relativa ganha nas multimodais ($F_6$, $F_7$, $F_9$), a absoluta ganha nas de vale
($F_2$, $F_4$, $F_{13}$). Se confirmar, o relatório tem um resultado de verdade e não só uma tabela melhor.

## O QUE JÁ EXISTE, E CONTRA O QUE ISSO SE POSICIONA

Honestidade sobre novidade é o que evita pergunta ruim na apresentação. O que já tem dono no slot do `F`:

| trabalho | o que faz | como a proposta difere |
| --- | --- | --- |
| Storn & Price (1997) | dither: um $F$ por geração | o dither é global; aqui é por indivíduo |
| jDE (Brest, 2006) | $F$ auto-adaptativo por indivíduo, **evoluído** junto com a solução | lá o $F$ é herdado e sorteado; aqui é **calculado** da posição atual |
| JADE / SHADE (2009, 2013) | $F$ de Cauchy em torno de média de sucessos | adaptação por histórico de sucesso; aqui não há histórico nenhum |
| IDE (Tang et al., 2015) | $F_i$ e $CR_i$ dependentes do **posto de aptidão** do indivíduo | **é o vizinho mais próximo**: mesma ideia de heterogeneizar a população, mas por aptidão, não por geometria |

A reivindicação é estreita e defensável: *escala diferencial por indivíduo derivada da distância ao
incumbente, com o normalizador como grau de liberdade do método*. Antes de escrever "inédita" no relatório,
conferir o slot no `Articles/08 DE 2022 A recent Review`.

## PROTOCOLO DE VALIDAÇÃO

**Triagem, antes de qualquer campanha completa.** Seis funções, só $D=30$, um terço do orçamento
($10^5$ avaliações), 5 sementes, as três versões (original, relativa, absoluta) com as **mesmas** sementes.
Custo: ~30 minutos. Critério para seguir: ganho em **4 das 6** e nenhuma piora clara. Se ficar em 2, troca-se
de fórmula enquanto trocar ainda é barato.

**Campanha final**, só depois da triagem aprovar: as duas versões (original + a variante vencedora) nas seis
funções, $D=30$ e $D=50$, 30 corridas, MaxFES $= 10^4 D$, com os dados oficiais de deslocamento e rotação do
CEC'14. São 24 campanhas de 30 corridas.

## O QUE ENTRA NO RELATÓRIO

- Tabela por função e dimensão no formato do CEC: melhor, pior, mediana, média, desvio — original contra modificada.
- **Wilcoxon signed-rank** por função (pareado, mesmas sementes) e **ranking de Friedman** sobre as doze campanhas.
- Curva de convergência: **erro mediano das 30 corridas** contra FES, em escala log, as duas versões sobrepostas.
- **Diversidade da população contra FES**, as duas versões sobrepostas — é a prova de que a fórmula faz o que
  a hipótese diz que ela faz, e não que o número melhorou por acaso.
- Comparação com a literatura recente: a L-SHADE já implementada no pacote é família campeã do CEC e serve
  de teto de referência na mesma tabela.

## RISCOS

1. **A borda vira ruído.** Com $F_i \to F_{max}$ nos mais distantes, os piores indivíduos podem virar busca
   aleatória cara. Mitigação: o $F_{max}$ é o mesmo teto do dither original, então o passo nunca é maior do
   que o que a DE clássica já usa.
2. **O $\max_j$ é uma estatística de ordem, portanto ruidosa.** Um único outlier na população define o
   normalizador da geração inteira. Alternativa a medir: quantil 0,9 ou média das distâncias.
3. **Convergência final na variante relativa.** É o risco conhecido e está declarado como hipótese — a
   variante absoluta existe justamente para cobrir esse caso.

## RESERVA

Se a triagem reprovar, o segundo candidato é trocar o **atrator** em vez da escala: substituir $x_{best}$
pela média dos $\mu$ melhores ponderada por posto, com os pesos logarítmicos da CMA-ES.

$$m_w = \sum_{k=1}^{\mu} w_k\,x_{(k)}, \qquad w_k \propto \ln(\mu + 0{,}5) - \ln k, \qquad \mu = N/2$$

$$v_i = x_i + F\,(m_w - x_i) + F\,(x_{r_1} - x_{r_2})$$

O $x_{best}$ sozinho é um atrator ruidoso — pode ser um ponto de sorte numa bacia ruim. A média ponderada por
posto é a direção de máxima verossimilhança de melhora, que é a razão pela qual a CMA-ES a usa, e é invariante
à rotação — que é exatamente o que essas seis funções cobram. Também é uma linha, num termo diferente da mesma
fórmula, e pode ser combinada com a escala geométrica depois que cada uma for medida sozinha.
