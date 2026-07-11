数据集来源：https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets

events数据集和markets数据集的关系：events是大主题，markets是每个主题下的具体预测场景
markets.event_id = events.id，一个event中包含多个markets

event样本：一共有4w条数据
主要属性有：
{
  "id": "35090",
  “category”: "Sports"  包含很多title，但是大多数据这个属性的值都为0
  "title": "Fed decision in December?", 基本上每条数据都有这个属性
  "slug": "fed-decision-in-december",  这条事件的“URL唯一标识”（单条事件级）
  "seriesSlug": "fed-interest-rates"  下面可以有很多slug的事件
  "active": "True",  是否当前可交易
  "closed": "False",
  "market_count": "4",  该事件下 market 数量
  "liquidity": "11838599.01708",  总流动性（事件层聚合）
  "volume": "225491875.945125",  总成交量
  "startDate": "2025-07-31T19:48:22.854077Z",
  "endDate": "2025-12-10T00:00:00Z"
}

market样本：一共有10w条数据
{
  "id": "570360",
  "event_id": "35090",
  "event_title": "Fed decision in December?",
  "question": "Fed decreases interest rates by 50+ bps after December 2025 meeting?",
  "slug": "fed-decreases-interest-rates-by-50-bps-after-december-2025-meeting",
  "active": "True",
  "closed": "False",
  "bestBid": "0.005",  当前市场里“最高买价”（买方愿意出的最高价格）。
  "bestAsk": "0.006",  当前市场里“最低卖价”（卖方愿意接受的最低价格）。
  "liquidity": "3558551.33955",   流动性强弱指标，数值越大，通常滑点越小、成交更平稳
  "volume": "73611752.066984",
  "outcomes": "[\"Yes\", \"No\"]",
  "outcomePrices": "[\"0.0055\", \"0.9945\"]"
}

抽数据：
1. event数据集中大部分数据都有title/slug/seriesSlug/tags/description这些属性，因此用这些属性给event分主题类，topic_classify_and_allocate.py
2. 再用market数据集每条数据的description属性，给每条数据分到1中这些主题类 build_final_market_dataset.py
3. 再按“类的受欢迎程度”（来自event热度分）给出每类应抽多少条market

Crypto: 31546条market，抽299
Sports: 20799条，抽110
ScienceTech: 27874条，抽45
Politics: 7572条，抽17
Business: 4946条，抽15
其余类合计：抽14

topic_market_summary.csv：每个主题的统计与建议配额（用于决策）。
market_topic_mapping.csv：每条 market 的主题映射明细（可审计）。
final_markets_500.csv：最终可用数据集（500 条，含 topic 与 quality_score）。
final_markets_500_stats.csv：最终 500 条在各主题下的实际数量对照。

证据采集逻辑（collect_evidences_tavily.py）
1. 输入：读取 outputs/final_markets_500.csv，按行处理 market（支持 --start-row 和 --limit 分段跑）。
2. 查询生成：每条 market 默认生成 3 条查询。
  - 有 LLM key：用问题、事件标题、描述生成 3 条扩展查询。
  - 无 LLM key：走启发式 fallback（question / question+event_title / question+description）。
3. 检索：每条查询调用 Tavily search（advanced，max_results=10）。
  - 如果该行有 endDate/endDateIso，会规范化为 YYYY-MM-DD 并作为 end_date 约束。
4. 去重与排序：按 url+title+内容片段做哈希去重；重复项保留 score 更高的结果，再按 score 降序排序。
5. 输出：每个 market 写一个 JSON 到 evidences/，文件名形如 row_0001_<market_id>.json。
  - 内容包含 expanded_queries、统计信息（raw/deduped）、evidences 列表和原始响应元信息。
6. 容错：已存在文件默认跳过（可 --overwrite）；单行失败会写 row_xxxx_ERROR.json，不影响后续行。

常用运行示例：
python data/scripts/collect_evidences_tavily.py \
  --input data/outputs/final_markets_500.csv \
  --output-dir data/evidences \
  --api-key $TAVILY_API_KEY \
  --llm-api-key $CMU_API_KEY