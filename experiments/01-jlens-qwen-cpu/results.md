# Jacobian-lens demo — Qwen/Qwen3.5-0.8B

- torch 2.12.1+cpu  (cuda available: False)
- transformers 5.13.0
- python 3.12.3
- lens: neuronpedia/jacobian-lens @ qwen-n1000 :: qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt

model loaded on CPU in 9s: HFLensModel(Qwen3_5ForCausalLM, n_layers=24, d_model=1024)
lens loaded: JacobianLens(d_model=1024, n_prompts=233, source_layers=[0..22] (23 layers))

n_layers=24, reading out at layers [6, 12, 18, 22] (+ model output at L23)
readout position = -1 (the token that predicts the answer), top-10

####################################################################################################
# PART 1 — readout at the ANSWER position (-1): lens vs model next-token
####################################################################################################

====================================================================================================
## Multi-hop currency (README/examples.py 'multihop')
prompt: 'Fact: The capital of Japan is Tokyo.\nFact: The currency used in the country shaped like a boot is'
hidden intermediate (not in prompt): 'Italy' | expected answer: 'euro / lira'
tokenized (23 tok), last 6: [' country', ' shaped', ' like', ' a', ' boot', ' is']
(forward+readout 3.0s)

  L 6 J-lens    : ' coins'  ' bark'  ' canvas'  ' barrel'  ' coin'  ' is'  ' cylindrical'  ' painting'  ' which'  ' cylinder'
  L 6 logit-lens: 'has'  ' ____'  'ieg'  '____'  'imo'  '___'  ' ______'  'omorphic'  'ometry'  '匿'

  L12 J-lens    : ' named'  ' coins'  ' name'  ' Indian'  '?'  ' Spanish'  ' Name'  ' coin'  ' Am'  ' Golden'
  L12 logit-lens: 'ありますか'  'mint'  ';**'  'manship'  'goed'  'ніка'  'مية'  ' Medik'  'vaard'  'abled'

  L18 J-lens    : ' currency'  ' currencies'  ' Currency'  '货币'  '人民币'  'Currency'  ' money'  ' coins'  ' Dollars'  ' Dollar'
  L18 logit-lens: ' currencies'  ' Currency'  ' currency'  'Currency'  'currency'  '货币'  '人民币'  ' moneda'  ' Dollars'  ' coins'

  L22 J-lens    : ' currency'  ' yen'  ' Yen'  ' currencies'  ' Currency'  ' Japanese'  ' coins'  ' USD'  ' coin'  '日元'
  L22 logit-lens: ' currency'  ' Currency'  '日元'  ' currencies'  'Currency'  'currency'  ' yen'  ' Coins'  '人民币'  ' coins'

  MODEL next-token (L23): ' the'  ' '  ' Japanese'  ' not'  ' J'  ' yen'  ' Yen'  ' a'  ' US'  ' $'

====================================================================================================
## Multi-hop Eiffel Tower currency
prompt: 'Fact: the currency of the country where the Eiffel Tower stands is'
hidden intermediate (not in prompt): 'France' | expected answer: 'euro / franc'
tokenized (14 tok), last 6: [' the', ' Eiff', 'el', ' Tower', ' stands', ' is']
(forward+readout 2.2s)

  L 6 J-lens    : ' country'  '—is'  ' coins'  ' islands'  ' Indies'  ' cities'  ' currency'  ' penn'  ' river'  ' coin'
  L 6 logit-lens: '会说'  'ộn'  'mét'  'has'  'ącz'  '导致了'  'hlas'  '匿'  '巡游'  '悲哀'

  L12 J-lens    : ' currency'  ' country'  ' Currency'  ' American'  ' Spanish'  ' nationality'  ' currencies'  ' Chile'  ' ____'  '?'
  L12 logit-lens: 'hlas'  ' ____'  'égal'  'étique'  'anguages'  ' обязаны'  '____'  ' nedir'  'banks'  ' Adap'

  L18 J-lens    : '欧元'  ' currency'  ' currencies'  ' Currency'  'Currency'  '人民币'  '货币'  '美元'  ' euro'  ' euros'
  L18 logit-lens: ' currencies'  '欧元'  ' Currency'  'Currency'  ' currency'  'currency'  ' euro'  '货币'  '英镑'  ' euros'

  L22 J-lens    : '欧元'  ' EUR'  ' euros'  ' euro'  ' French'  ' Euro'  ' Eiff'  ' franc'  ' currency'  ' Franc'
  L22 logit-lens: '欧元'  ' franc'  ' euros'  ' euro'  ' not'  ' currency'  ' Currency'  ' called'  '外币'  ' EUR'

  MODEL next-token (L23): ' the'  ' French'  ' not'  ' in'  '\n'  '<|im_end|>'  ' '  ' called'  ' a'  ' France'

====================================================================================================
## Multi-hop Carnival ocean (data/evaluations/lens-eval-multihop 'carnival-ocean')
prompt: 'Fact: The ocean on the coast of the country where Carnival is most famously celebrated is the'
hidden intermediate (not in prompt): 'Brazil' | expected answer: 'Atlantic'
tokenized (18 tok), last 6: [' is', ' most', ' famously', ' celebrated', ' is', ' the']
(forward+readout 2.4s)

  L 6 J-lens    : ' Americans'  ' American'  ' coast'  ' north'  ' south'  ' east'  ' sea'  ' ocean'  ' continent'  ' Asia'
  L 6 logit-lens: ' entire'  ' –'  ' technically'  ' '  '\\n'  ' most'  '像'  ' closest'  '应该'  '现在'

  L12 J-lens    : ' river'  ' continent'  ' south'  ' north'  ' Americas'  ' plains'  ' sea'  '?\\'  ' ___'  ' ocean'
  L12 logit-lens: ' celebrated'  'voie'  ' names'  ' entire'  ' same'  ' reserves'  ' ___'  'Languages'  '？'  '..."'

  L18 J-lens    : ' ocean'  ' Ocean'  ' oceans'  ' Sea'  ' sea'  ' seas'  'Ocean'  ' coastline'  '海洋'  ' waters'
  L18 logit-lens: ' oceans'  ' ocean'  ' sea'  ' seas'  ' Sea'  ' largest'  ' waters'  ' entirety'  ' Atlantic'  ' biggest'

  L22 J-lens    : ' Caribbean'  ' Atlantic'  ' ocean'  ' Pacific'  ' largest'  ' Mediterranean'  ' sea'  ' Ocean'  ' Sea'  ' Gulf'
  L22 logit-lens: ' largest'  ' biggest'  ' same'  ' entirety'  ' Caribbean'  ' Atlantic'  ' ocean'  ' tallest'  ' hottest'  ' only'

  MODEL next-token (L23): ' Caribbean'  ' Atlantic'  ' Pacific'  ' largest'  ' ocean'  '\n'  ' only'  ' **'  ' world'  ' most'

####################################################################################################
# PART 2 — does the hidden intermediate surface at the descriptor token?
####################################################################################################

====================================================================================================
## Multi-hop currency (README/examples.py 'multihop')
prompt: 'Fact: The capital of Japan is Tokyo.\nFact: The currency used in the country shaped like a boot is'
descriptor token = ' boot' (index 21); hidden intermediate = 'Italy' (token 14898 = ' Italy')

  best lens rank of 'Italy' over all layers at that token: rank 5792 at L15

  L 6 @      ' boot'  [Italy rank 10995]: ' boots'  ' boot'  '靴'  '-boot'  ' heel'  'boot'  ' shoe'  ' ankle'  ' shoes'  '/boot'
  L12 @      ' boot'  [Italy rank 6582]: ' boots'  ' boot'  ' heel'  ' shoes'  ' shoe'  '靴'  ' legs'  'legs'  'heel'  'boot'
  L15 @      ' boot'  [Italy rank 5792]: ' boots'  'legs'  '靴'  'heel'  ' ankles'  ' boot'  '-legged'  ' heel'  '-boot'  ' legs'
  L18 @      ' boot'  [Italy rank 39540]: ' boots'  '靴'  'heel'  'legs'  'boot'  ' boot'  '-boot'  'Boot'  ' heel'  ' shoes'

====================================================================================================
## Multi-hop Eiffel Tower currency
prompt: 'Fact: the currency of the country where the Eiffel Tower stands is'
descriptor token = ' Tower' (index 11); hidden intermediate = 'France' (token 9338 = ' France')

  best lens rank of 'France' over all layers at that token: rank 100 at L21

  L 6 @     ' Tower'  [France rank 1573]: ' Tower'  ' tower'  ' towers'  '铁塔'  ' Eiff'  ' Towers'  'tower'  'Tower'  ' skys'  '塔的'
  L12 @     ' Tower'  [France rank 433]: ' Tower'  ' tower'  ' towers'  ' Eiff'  ' Towers'  ' Statue'  ' skys'  'tower'  ' statue'  '铁塔'
  L21 @     ' Tower'  [France rank 100]: ' is'  ' was'  ' sits'  ' stands'  ' Eiff'  ' headquarters'  ' resides'  ' operates'  ' stood'  ' exists'
  L18 @     ' Tower'  [France rank 5883]: ' Tower'  ' towers'  ' tower'  ' sits'  ' buildings'  ' erected'  ' Buildings'  '是一座'  ' Towers'  ' skys'

====================================================================================================
## Multi-hop Carnival ocean (data/evaluations/lens-eval-multihop 'carnival-ocean')
prompt: 'Fact: The ocean on the coast of the country where Carnival is most famously celebrated is the'
descriptor token = ' Carnival' (index 11); hidden intermediate = 'Brazil' (token 15477 = ' Brazil')

  best lens rank of 'Brazil' over all layers at that token: rank 338 at L12

  L 6 @  ' Carnival'  [Brazil rank 1055]: ' Carnival'  ' carnival'  ' Cruise'  ' Mardi'  ' cruise'  ' Festival'  ' festival'  '嘉年华'  ' nightclub'  ' Circus'
  L12 @  ' Carnival'  [Brazil rank 338]: ' Carnival'  ' carnival'  ' Cruise'  ' Festival'  ' Mardi'  ' Circus'  ' cruise'  ' festival'  ' Fiesta'  ' nightclub'
  L12 @  ' Carnival'  [Brazil rank 338]: ' Carnival'  ' carnival'  ' Cruise'  ' Festival'  ' Mardi'  ' Circus'  ' cruise'  ' festival'  ' Fiesta'  ' nightclub'
  L18 @  ' Carnival'  [Brazil rank 6777]: ' Carnival'  ' Cruise'  ' Parade'  ' Celebration'  ' Festival'  ' carnival'  ' Ships'  ' Fiesta'  ' Tickets'  ' Circus'

