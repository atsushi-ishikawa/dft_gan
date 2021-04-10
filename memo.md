* Generator lossがDiscriminator lossより大きいほうがうまくいく印象。G-lossが4、D-lossが2くらいがいいような気がする
* でもG-lossが大きいと(5くらい)予測ステップがあまり上手ではない
* DOSの生成とatomic numberの生成に適したarchtechtureはちがうような気がする。DOSの場合は、BatchNormが邪魔をするがatomic numberではBatchNormがあると非常に良い
* うまくいってる時のloss
* generatorのparameter数が小さいと"より絞り込んだ"感じになり、parameterが多いとgenerated sample数も多くなる気がする

<img src=file:///Users/ishi/ase/nn_reac/results/loss200612.png width=50%>

* うまくいっている時の予測図

<img src=file:///Users/ishi/ase/nn_reac/results/predictions_200612.png width=50%>

