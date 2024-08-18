ディレクトリ構成
- game: pygameで実際に操作できる
- main.pyで実行

- train_ac_ppo.pyで学習を実行
- try.pyで学習モデルを実行

- try_new.pyでPPO(from stable_baselines3)学習
- ミサイル対応ゲーム=_missileがついたファイル
- env最新はenv_new_4

# 模倣学習
- try_new_play.pyでランダムに動く敵と対戦しながらエキスパートデータ生成
- try_new_bc.pyでbehavior cloningで学習
- try_new_airl.pyでAIRLで学習
- try_new_test.pyで学習されたモデルの挙動テスト
- try_new_play_part2.pyで、学習したAIと対戦しながらエキスパートデータ生成
