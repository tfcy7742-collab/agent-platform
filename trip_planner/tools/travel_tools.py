"""旅行工具函数集合（模拟数据实现）。

本模块提供三个工具，供各智能体通过 Function Calling 调用：

1. ``search_attractions(destination, preference)`` —— 搜索景点
2. ``get_weather(destination, dates)``             —— 查询天气
3. ``search_hotels(destination, budget_level)``    —— 搜索酒店

设计说明
--------
* 全部使用本地 Python 字典模拟数据，**不依赖任何真实第三方 API**，因此项目开箱即跑。
* 数据覆盖 北京 / 上海 / 成都 / 西安 / 杭州 共 5 个热门目的地（满足"至少 3 个"的要求）。
* 未知城市会走"通用兜底逻辑"：用城市名确定性生成可复现的数据，
  保证接口永远不会返回空结果。
* 每个工具函数都有 ``*_tool_spec()`` 形式的 JSON Schema 描述，
  可直接喂给 DeepSeek 的 function calling。
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any, Dict, List

from trip_planner.models.schemas import Attraction, Hotel, WeatherInfo

# ---------------------------------------------------------------------------
# 一、模拟数据库：景点
# ---------------------------------------------------------------------------
# 每个景点包含：名称、简介、建议游览时长（小时）、门票（元）、评分、标签、最佳时间、所在区域
ATTRACTIONS_DB: Dict[str, List[Dict[str, Any]]] = {
    "北京": [
        {
            "name": "故宫博物院",
            "description": "明清两代皇家宫殿，世界现存规模最大的木质结构古建筑群，中轴线核心。",
            "duration_hours": 4.0,
            "ticket_price": 60,
            "rating": 4.9,
            "tags": ["历史文化", "摄影", "亲子"],
            "best_time": "全年（周一闭馆）",
            "location": "东城区景山前街",
        },
        {
            "name": "八达岭长城",
            "description": "明长城保存最完好的地段，登高可俯瞰燕山群峰，是「不到长城非好汉」的打卡地。",
            "duration_hours": 4.5,
            "ticket_price": 40,
            "rating": 4.8,
            "tags": ["历史文化", "自然风光", "摄影"],
            "best_time": "4-10 月",
            "location": "延庆区",
        },
        {
            "name": "颐和园",
            "description": "清代皇家园林，昆明湖与万寿山构成「一池三山」格局，长廊彩绘精美。",
            "duration_hours": 3.0,
            "ticket_price": 30,
            "rating": 4.7,
            "tags": ["历史文化", "自然风光", "休闲度假"],
            "best_time": "4-10 月",
            "location": "海淀区新建宫门路",
        },
        {
            "name": "天坛公园",
            "description": "明清皇帝祭天祈谷之所，祈年殿为中国古代建筑美学巅峰。",
            "duration_hours": 2.5,
            "ticket_price": 34,
            "rating": 4.6,
            "tags": ["历史文化", "亲子"],
            "best_time": "全年",
            "location": "东城区天坛内东里",
        },
        {
            "name": "南锣鼓巷与什刹海",
            "description": "老北京胡同肌理保留区，可骑行、划船、逛小店，夜景与酒吧街颇具氛围。",
            "duration_hours": 3.0,
            "ticket_price": 0,
            "rating": 4.5,
            "tags": ["美食", "夜生活", "购物", "摄影"],
            "best_time": "傍晚至夜间",
            "location": "东城区 / 西城区",
        },
        {
            "name": "798 艺术区",
            "description": "由老厂房改造的当代艺术园区，画廊、展览、咖啡馆与工业遗存交织。",
            "duration_hours": 2.5,
            "ticket_price": 0,
            "rating": 4.4,
            "tags": ["摄影", "购物", "休闲度假"],
            "best_time": "全年",
            "location": "朝阳区酒仙桥",
        },
        {
            "name": "北京环球度假区",
            "description": "大型主题乐园，含哈利·波特、变形金刚等七大主题景区。",
            "duration_hours": 8.0,
            "ticket_price": 528,
            "rating": 4.7,
            "tags": ["亲子", "休闲度假"],
            "best_time": "全年",
            "location": "通州区",
        },
        {
            "name": "雍和宫与国子监",
            "description": "京城规模最大的藏传佛教寺院，毗邻元明清三代最高学府国子监。",
            "duration_hours": 2.0,
            "ticket_price": 25,
            "rating": 4.6,
            "tags": ["历史文化"],
            "best_time": "全年",
            "location": "东城区雍和宫大街",
        },
    ],
    "上海": [
        {
            "name": "外滩与陆家嘴天际线",
            "description": "万国建筑博览群与对岸摩天大楼隔江相望，是上海最具标志性的城市景观。",
            "duration_hours": 2.5,
            "ticket_price": 0,
            "rating": 4.8,
            "tags": ["摄影", "夜生活", "购物"],
            "best_time": "傍晚至夜间",
            "location": "黄浦区中山东一路",
        },
        {
            "name": "豫园与城隍庙",
            "description": "明代江南古典园林，周边老城厢小吃街汇聚南翔小笼、梨膏糖等老字号。",
            "duration_hours": 3.0,
            "ticket_price": 40,
            "rating": 4.5,
            "tags": ["历史文化", "美食", "购物"],
            "best_time": "全年",
            "location": "黄浦区安仁街",
        },
        {
            "name": "上海迪士尼乐园",
            "description": "中国内地首座迪士尼主题乐园，含七大主题园区与夜光幻影秀。",
            "duration_hours": 9.0,
            "ticket_price": 545,
            "rating": 4.7,
            "tags": ["亲子", "休闲度假"],
            "best_time": "全年",
            "location": "浦东新区川沙",
        },
        {
            "name": "武康路与安福路街区",
            "description": "梧桐树下的历史风貌街区，老洋房、买手店与咖啡馆密度极高。",
            "duration_hours": 2.5,
            "ticket_price": 0,
            "rating": 4.6,
            "tags": ["摄影", "购物", "休闲度假"],
            "best_time": "春秋两季",
            "location": "徐汇区",
        },
        {
            "name": "上海博物馆（东馆）",
            "description": "中国古代艺术博物馆，青铜器、陶瓷、书画馆藏为国内顶级。",
            "duration_hours": 3.0,
            "ticket_price": 0,
            "rating": 4.8,
            "tags": ["历史文化", "亲子"],
            "best_time": "全年（周一闭馆）",
            "location": "浦东新区世纪大道",
        },
        {
            "name": "田子坊与新天地",
            "description": "石库门里弄改造的创意街区，餐饮、酒吧与设计小店聚集。",
            "duration_hours": 2.5,
            "ticket_price": 0,
            "rating": 4.4,
            "tags": ["美食", "夜生活", "购物"],
            "best_time": "傍晚至夜间",
            "location": "黄浦区泰康路 / 太仓路",
        },
        {
            "name": "上海中心大厦观光厅",
            "description": "632 米高中国第一高楼，118 层观光厅可 360° 俯瞰浦江两岸。",
            "duration_hours": 1.5,
            "ticket_price": 180,
            "rating": 4.6,
            "tags": ["摄影", "亲子"],
            "best_time": "晴天日落时分",
            "location": "浦东新区银城中路",
        },
    ],
    "成都": [
        {
            "name": "成都大熊猫繁育研究基地",
            "description": "全球最大的大熊猫迁地保护基地，清晨是熊猫最活跃的时段。",
            "duration_hours": 3.5,
            "ticket_price": 55,
            "rating": 4.8,
            "tags": ["亲子", "自然风光", "摄影"],
            "best_time": "全年（建议 8:00 前入园）",
            "location": "成华区熊猫大道",
        },
        {
            "name": "宽窄巷子",
            "description": "清代古街巷改造的休闲街区，盖碗茶、川剧变脸与成都小吃齐聚。",
            "duration_hours": 2.5,
            "ticket_price": 0,
            "rating": 4.5,
            "tags": ["历史文化", "美食", "购物"],
            "best_time": "全年",
            "location": "青羊区长顺上街",
        },
        {
            "name": "武侯祠与锦里",
            "description": "纪念诸葛亮的君臣合祀祠庙，相邻锦里古街夜间红灯笼极具年味。",
            "duration_hours": 3.0,
            "ticket_price": 50,
            "rating": 4.6,
            "tags": ["历史文化", "美食", "夜生活"],
            "best_time": "傍晚至夜间",
            "location": "武侯区武侯祠大街",
        },
        {
            "name": "都江堰景区",
            "description": "战国时期李冰主持修建的无坝引水工程，两千余年仍在灌溉成都平原。",
            "duration_hours": 4.0,
            "ticket_price": 80,
            "rating": 4.7,
            "tags": ["历史文化", "自然风光"],
            "best_time": "3-11 月",
            "location": "都江堰市",
        },
        {
            "name": "青城山",
            "description": "中国道教发源地之一，「青城天下幽」，前山道观密集、后山溪谷清幽。",
            "duration_hours": 5.0,
            "ticket_price": 80,
            "rating": 4.7,
            "tags": ["自然风光", "历史文化", "休闲度假"],
            "best_time": "3-11 月",
            "location": "都江堰市青城山镇",
        },
        {
            "name": "人民公园鹤鸣茶社",
            "description": "百年老茶馆，竹椅盖碗茶配采耳，是体验成都慢生活的最佳去处。",
            "duration_hours": 2.0,
            "ticket_price": 0,
            "rating": 4.6,
            "tags": ["美食", "休闲度假"],
            "best_time": "全天",
            "location": "青羊区少城路",
        },
        {
            "name": "春熙路与太古里",
            "description": "成都最繁华的商圈，IFS 熊猫爬楼雕塑是必拍机位。",
            "duration_hours": 2.5,
            "ticket_price": 0,
            "rating": 4.5,
            "tags": ["购物", "美食", "夜生活"],
            "best_time": "傍晚至夜间",
            "location": "锦江区",
        },
    ],
    "西安": [
        {
            "name": "秦始皇兵马俑博物院",
            "description": "世界第八大奇迹，三个俑坑展现秦代军阵与陶塑工艺。",
            "duration_hours": 4.0,
            "ticket_price": 120,
            "rating": 4.9,
            "tags": ["历史文化", "亲子"],
            "best_time": "全年",
            "location": "临潼区",
        },
        {
            "name": "西安城墙",
            "description": "中国现存规模最大的古代城垣，可租自行车环城骑行 13.7 公里。",
            "duration_hours": 2.5,
            "ticket_price": 54,
            "rating": 4.7,
            "tags": ["历史文化", "摄影", "亲子"],
            "best_time": "傍晚",
            "location": "碑林区",
        },
        {
            "name": "大雁塔与大唐不夜城",
            "description": "玄奘译经之地，夜间不夜城灯光演艺与不倒翁小姐姐人气极高。",
            "duration_hours": 3.0,
            "ticket_price": 50,
            "rating": 4.7,
            "tags": ["历史文化", "夜生活", "摄影"],
            "best_time": "傍晚至夜间",
            "location": "雁塔区",
        },
        {
            "name": "回民街与永兴坊",
            "description": "西安小吃集中地，肉夹馍、羊肉泡馍、摔碗酒一网打尽。",
            "duration_hours": 2.0,
            "ticket_price": 0,
            "rating": 4.4,
            "tags": ["美食", "夜生活"],
            "best_time": "傍晚至夜间",
            "location": "莲湖区",
        },
        {
            "name": "陕西历史博物馆",
            "description": "被誉为「古都明珠」，何家村窖藏金银器与唐墓壁画馆为镇馆之宝。",
            "duration_hours": 3.0,
            "ticket_price": 0,
            "rating": 4.8,
            "tags": ["历史文化", "亲子"],
            "best_time": "全年（周一闭馆，需预约）",
            "location": "雁塔区小寨东路",
        },
    ],
    "杭州": [
        {
            "name": "西湖风景名胜区",
            "description": "世界文化遗产，「一山二塔三岛三堤五湖」格局，四季皆宜。",
            "duration_hours": 4.0,
            "ticket_price": 0,
            "rating": 4.9,
            "tags": ["自然风光", "摄影", "休闲度假"],
            "best_time": "3-5 月、9-11 月",
            "location": "西湖区",
        },
        {
            "name": "灵隐寺与飞来峰",
            "description": "江南著名古刹，飞来峰五代宋元造像群为全国重点文物。",
            "duration_hours": 3.0,
            "ticket_price": 75,
            "rating": 4.7,
            "tags": ["历史文化", "自然风光"],
            "best_time": "全年",
            "location": "西湖区法云弄",
        },
        {
            "name": "西溪国家湿地公园",
            "description": "城市湿地，「一曲溪流一曲烟」，可摇橹船深入芦苇荡。",
            "duration_hours": 3.5,
            "ticket_price": 80,
            "rating": 4.6,
            "tags": ["自然风光", "亲子", "休闲度假"],
            "best_time": "3-11 月",
            "location": "西湖区天目山路",
        },
        {
            "name": "河坊街与南宋御街",
            "description": "杭州老城历史街区，胡庆余堂、王星记扇庄等老字号云集。",
            "duration_hours": 2.5,
            "ticket_price": 0,
            "rating": 4.4,
            "tags": ["美食", "购物", "历史文化"],
            "best_time": "傍晚至夜间",
            "location": "上城区",
        },
        {
            "name": "龙井村茶园",
            "description": "西湖龙井核心产区，可体验采茶、炒茶与茶山徒步。",
            "duration_hours": 3.0,
            "ticket_price": 0,
            "rating": 4.6,
            "tags": ["自然风光", "休闲度假", "摄影"],
            "best_time": "3-5 月",
            "location": "西湖区龙井村",
        },
    ],
}

# ---------------------------------------------------------------------------
# 二、模拟数据库：酒店
# ---------------------------------------------------------------------------
HOTELS_DB: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "北京": {
        "经济": [
            {
                "name": "如家精选酒店（前门大栅栏店）",
                "price_per_night": 328,
                "rating": 4.4,
                "location": "西城区大栅栏商业街",
                "level": "经济",
                "tags": ["近地铁", "免费 Wi-Fi"],
                "distance_to_center": "距天安门约 1.5km",
            },
            {
                "name": "汉庭酒店（王府井店）",
                "price_per_night": 359,
                "rating": 4.3,
                "location": "东城区王府井大街",
                "level": "经济",
                "tags": ["近地铁", "24h 前台"],
                "distance_to_center": "距天安门约 2.0km",
            },
            {
                "name": "7 天优品（鼓楼店）",
                "price_per_night": 289,
                "rating": 4.2,
                "location": "东城区鼓楼东大街",
                "level": "经济",
                "tags": ["胡同风情", "性价比高"],
                "distance_to_center": "距什刹海约 0.8km",
            },
        ],
        "中等": [
            {
                "name": "北京王府井希尔顿花园酒店",
                "price_per_night": 828,
                "rating": 4.7,
                "location": "东城区王府井大街",
                "level": "中等",
                "tags": ["近地铁", "健身房", "含早"],
                "distance_to_center": "距天安门约 1.8km",
            },
            {
                "name": "桔子水晶北京故宫酒店",
                "price_per_night": 698,
                "rating": 4.6,
                "location": "东城区南河沿大街",
                "level": "中等",
                "tags": ["设计感", "近故宫", "含早"],
                "distance_to_center": "距故宫约 1.2km",
            },
            {
                "name": "北京国贸大酒店（行政公寓）",
                "price_per_night": 758,
                "rating": 4.5,
                "location": "朝阳区建国门外大街",
                "level": "中等",
                "tags": ["商圈核心", "商务出行"],
                "distance_to_center": "距天安门约 6.0km",
            },
        ],
        "豪华": [
            {
                "name": "北京王府半岛酒店",
                "price_per_night": 2380,
                "rating": 4.9,
                "location": "东城区金鱼胡同",
                "level": "豪华",
                "tags": ["奢华", "米其林餐厅", "SPA"],
                "distance_to_center": "距故宫约 1.5km",
            },
            {
                "name": "北京颐和安缦",
                "price_per_night": 4680,
                "rating": 4.9,
                "location": "海淀区颐和园东宫门",
                "level": "豪华",
                "tags": ["皇家园林", "私汤", "静谧"],
                "distance_to_center": "紧邻颐和园",
            },
            {
                "name": "北京华尔道夫酒店",
                "price_per_night": 1980,
                "rating": 4.8,
                "location": "东城区金鱼胡同",
                "level": "豪华",
                "tags": ["四合院", "管家服务"],
                "distance_to_center": "距王府井约 0.5km",
            },
        ],
    },
    "上海": {
        "经济": [
            {
                "name": "如家精选酒店（人民广场店）",
                "price_per_night": 379,
                "rating": 4.4,
                "location": "黄浦区南京西路",
                "level": "经济",
                "tags": ["近地铁", "交通便利"],
                "distance_to_center": "距外滩约 1.5km",
            },
            {
                "name": "锦江之星（豫园店）",
                "price_per_night": 329,
                "rating": 4.3,
                "location": "黄浦区人民路",
                "level": "经济",
                "tags": ["老城厢", "美食环绕"],
                "distance_to_center": "距外滩约 1.0km",
            },
        ],
        "中等": [
            {
                "name": "上海外滩华尔道夫酒店（精选房型）",
                "price_per_night": 1180,
                "rating": 4.8,
                "location": "黄浦区中山东一路",
                "level": "中等",
                "tags": ["江景", "历史建筑"],
                "distance_to_center": "外滩核心",
            },
            {
                "name": "亚朵酒店（新天地店）",
                "price_per_night": 748,
                "rating": 4.6,
                "location": "黄浦区马当路",
                "level": "中等",
                "tags": ["设计感", "含早", "近地铁"],
                "distance_to_center": "距外滩约 2.5km",
            },
            {
                "name": "上海静安瑞吉酒店（行政房）",
                "price_per_night": 988,
                "rating": 4.7,
                "location": "静安区北京西路",
                "level": "中等",
                "tags": ["商务", "健身房"],
                "distance_to_center": "距人民广场约 1.8km",
            },
        ],
        "豪华": [
            {
                "name": "上海和平饭店",
                "price_per_night": 2280,
                "rating": 4.9,
                "location": "黄浦区南京东路",
                "level": "豪华",
                "tags": ["历史地标", "爵士酒吧"],
                "distance_to_center": "外滩核心",
            },
            {
                "name": "上海浦东丽思卡尔顿酒店",
                "price_per_night": 2680,
                "rating": 4.9,
                "location": "浦东新区陆家嘴",
                "level": "豪华",
                "tags": ["江景", "天际泳池"],
                "distance_to_center": "陆家嘴核心",
            },
        ],
    },
    "成都": {
        "经济": [
            {
                "name": "汉庭酒店（春熙路店）",
                "price_per_night": 259,
                "rating": 4.4,
                "location": "锦江区红星路",
                "level": "经济",
                "tags": ["商圈核心", "近地铁"],
                "distance_to_center": "距天府广场约 1.0km",
            },
            {
                "name": "如家酒店（宽窄巷子店）",
                "price_per_night": 289,
                "rating": 4.3,
                "location": "青羊区同仁路",
                "level": "经济",
                "tags": ["近景区", "小吃环绕"],
                "distance_to_center": "距宽窄巷子约 0.5km",
            },
        ],
        "中等": [
            {
                "name": "成都太古里亚朵酒店",
                "price_per_night": 668,
                "rating": 4.7,
                "location": "锦江区中纱帽街",
                "level": "中等",
                "tags": ["设计感", "含早", "商圈核心"],
                "distance_to_center": "距春熙路约 0.3km",
            },
            {
                "name": "成都香格里拉大酒店（豪华阁）",
                "price_per_night": 898,
                "rating": 4.6,
                "location": "武侯区滨江东路",
                "level": "中等",
                "tags": ["江景", "行政酒廊"],
                "distance_to_center": "距天府广场约 3.0km",
            },
        ],
        "豪华": [
            {
                "name": "成都博舍酒店",
                "price_per_night": 2180,
                "rating": 4.9,
                "location": "锦江区笔帖式街",
                "level": "豪华",
                "tags": ["川西院落", "极简设计"],
                "distance_to_center": "太古里核心",
            },
            {
                "name": "成都华尔道夫酒店",
                "price_per_night": 1880,
                "rating": 4.8,
                "location": "武侯区天府大道",
                "level": "豪华",
                "tags": ["高层城景", "SPA"],
                "distance_to_center": "距天府广场约 5.0km",
            },
        ],
    },
    "西安": {
        "经济": [
            {
                "name": "汉庭酒店（钟楼店）",
                "price_per_night": 239,
                "rating": 4.3,
                "location": "碑林区东大街",
                "level": "经济",
                "tags": ["近钟楼", "交通便利"],
                "distance_to_center": "距钟楼约 0.6km",
            }
        ],
        "中等": [
            {
                "name": "西安钟楼索菲特传奇酒店",
                "price_per_night": 698,
                "rating": 4.6,
                "location": "碑林区东大街",
                "level": "中等",
                "tags": ["近城墙", "含早"],
                "distance_to_center": "距钟楼约 0.8km",
            }
        ],
        "豪华": [
            {
                "name": "西安 W 酒店",
                "price_per_night": 1580,
                "rating": 4.8,
                "location": "雁塔区雁南四路",
                "level": "豪华",
                "tags": ["潮流设计", "泳池"],
                "distance_to_center": "距大雁塔约 2.0km",
            }
        ],
    },
    "杭州": {
        "经济": [
            {
                "name": "如家精选酒店（西湖店）",
                "price_per_night": 299,
                "rating": 4.4,
                "location": "上城区解放路",
                "level": "经济",
                "tags": ["近西湖", "近地铁"],
                "distance_to_center": "距西湖约 1.0km",
            }
        ],
        "中等": [
            {
                "name": "杭州西湖柳莺里酒店",
                "price_per_night": 1080,
                "rating": 4.8,
                "location": "西湖区南山路",
                "level": "中等",
                "tags": ["湖景", "园林"],
                "distance_to_center": "紧邻西湖",
            },
            {
                "name": "亚朵酒店（武林广场店）",
                "price_per_night": 628,
                "rating": 4.6,
                "location": "拱墅区体育场路",
                "level": "中等",
                "tags": ["含早", "商务"],
                "distance_to_center": "距西湖约 2.5km",
            },
        ],
        "豪华": [
            {
                "name": "杭州西子湖四季酒店",
                "price_per_night": 3280,
                "rating": 4.9,
                "location": "西湖区灵隐路",
                "level": "豪华",
                "tags": ["湖景园林", "SPA"],
                "distance_to_center": "紧邻西湖",
            }
        ],
    },
}

# ---------------------------------------------------------------------------
# 三、模拟数据库：气候基线（用于确定性生成天气）
# ---------------------------------------------------------------------------
# 结构：城市 -> 12 个月的平均 (最低温, 最高温)
CLIMATE_BASE: Dict[str, List[tuple]] = {
    "北京": [(-7, 2), (-5, 5), (1, 12), (8, 20), (14, 26), (19, 30),
             (22, 31), (21, 30), (15, 25), (8, 19), (0, 10), (-5, 3)],
    "上海": [(1, 8), (3, 10), (7, 14), (12, 20), (17, 25), (21, 29),
             (25, 33), (25, 32), (21, 28), (15, 23), (9, 17), (3, 11)],
    "成都": [(3, 10), (5, 12), (9, 17), (14, 23), (18, 27), (21, 30),
             (23, 32), (23, 32), (19, 27), (15, 21), (9, 16), (4, 11)],
    "西安": [(-4, 5), (-1, 8), (4, 14), (10, 21), (15, 26), (20, 32),
             (23, 33), (22, 31), (17, 26), (11, 20), (4, 13), (-2, 6)],
    "杭州": [(1, 8), (3, 10), (7, 15), (12, 21), (18, 26), (22, 29),
             (25, 34), (25, 33), (21, 28), (15, 23), (9, 17), (3, 11)],
}

# 天气状况候选与对应的出行建议
WEATHER_CONDITIONS: List[Dict[str, str]] = [
    {"condition": "晴", "suggestion": "天气晴好，注意防晒并随身补水，适合户外景点。"},
    {"condition": "多云", "suggestion": "体感舒适，适合全天户外活动，早晚建议加一件外套。"},
    {"condition": "阴", "suggestion": "光线柔和适合拍照，博物馆类室内景点体验更佳。"},
    {"condition": "小雨", "suggestion": "请携带雨具，建议把室内展馆安排在降雨时段。"},
    {"condition": "阵雨", "suggestion": "降雨间歇较短，随身带伞并预留室内备选行程。"},
    {"condition": "雷阵雨", "suggestion": "午后易有雷阵雨，避免登长城、爬山等露天高处活动。"},
]

WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


# ---------------------------------------------------------------------------
# 工具内部辅助函数
# ---------------------------------------------------------------------------
def _stable_seed(*parts: str) -> int:
    """根据输入生成稳定的整数种子，保证同一输入得到同一份模拟数据（可复现）。"""
    raw = "|".join(parts).encode("utf-8")
    return int(hashlib.md5(raw).hexdigest(), 16)


def _weather_fallback(destination: str, dates: List[str]) -> List[Dict[str, Any]]:
    """未知城市时，用基线气候 + 城市名哈希生成确定性的天气数据。"""
    results: List[Dict[str, Any]] = []
    for date_str in dates:
        try:
            day = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            # 容错：非法日期退化为今天
            day = date.today()
        seed = _stable_seed(destination, date_str)
        # 用北京气候作为基准，并按城市名偏移，保证不同城市数据不同
        base = CLIMATE_BASE["北京"][day.month - 1]
        offset = (seed % 7) - 3
        cond = WEATHER_CONDITIONS[seed % len(WEATHER_CONDITIONS)]
        results.append(
            {
                "date": day.isoformat(),
                "weekday": WEEKDAY_CN[day.weekday()],
                "condition": cond["condition"],
                "temp_min": base[0] + offset,
                "temp_max": base[1] + offset,
                "wind": f"{2 + seed % 4} 级",
                "suggestion": cond["suggestion"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# 四、对外工具函数
# ---------------------------------------------------------------------------
def search_attractions(destination: str, preference: str = "", limit: int = 8) -> List[Dict[str, Any]]:
    """根据目的地与偏好搜索景点。

    Args:
        destination: 目的地城市，例如 "北京"。
        preference: 偏好关键词，可以是 "历史文化"，也可以是 "历史文化,美食"。
        limit: 最多返回多少个景点。

    Returns:
        景点字典列表（已按"偏好匹配度 + 评分"排序）。
        未知城市时返回通用城市景点（由城市名确定性生成），不会返回空列表。
    """
    destination = (destination or "").strip()
    prefs = [p.strip() for p in (preference or "").replace("，", ",").split(",") if p.strip()]

    raw_list = ATTRACTIONS_DB.get(destination)
    if raw_list is None:
        # 未知城市兜底：构造一份通用景点，仍然保证名称里带城市名，便于前端展示
        raw_list = [
            {
                "name": f"{destination}城市地标广场",
                "description": f"{destination}最具代表性的城市中心广场，适合作为行程起点与夜间散步。",
                "duration_hours": 2.0,
                "ticket_price": 0,
                "rating": 4.5,
                "tags": ["摄影", "休闲度假"],
                "best_time": "傍晚",
                "location": f"{destination}市中心",
            },
            {
                "name": f"{destination}历史博物馆",
                "description": f"系统化了解{destination}历史沿革与地方文化的首选室内场馆。",
                "duration_hours": 2.5,
                "ticket_price": 0,
                "rating": 4.6,
                "tags": ["历史文化", "亲子"],
                "best_time": "全年",
                "location": f"{destination}文化区",
            },
            {
                "name": f"{destination}老街美食街",
                "description": f"汇集{destination}本地小吃与老字号，适合安排一顿地道晚餐。",
                "duration_hours": 2.0,
                "ticket_price": 0,
                "rating": 4.4,
                "tags": ["美食", "夜生活"],
                "best_time": "夜间",
                "location": f"{destination}老城区",
            },
            {
                "name": f"{destination}城市公园",
                "description": f"{destination}市民休闲绿地，适合晨练、散步与亲子活动。",
                "duration_hours": 1.5,
                "ticket_price": 0,
                "rating": 4.3,
                "tags": ["自然风光", "亲子", "休闲度假"],
                "best_time": "清晨",
                "location": f"{destination}近郊",
            },
            {
                "name": f"{destination}滨水风光带",
                "description": f"沿江/沿海步道，夜景灯光与城市天际线视野极佳。",
                "duration_hours": 2.0,
                "ticket_price": 0,
                "rating": 4.5,
                "tags": ["自然风光", "摄影", "夜生活"],
                "best_time": "日落时分",
                "location": f"{destination}滨水区",
            },
        ]

    scored: List[tuple] = []
    for item in raw_list:
        # 偏好匹配度：命中一个偏好 +10 分；未指定偏好时全部平等
        hit = len(set(prefs) & set(item.get("tags", [])))
        score = hit * 10 + float(item.get("rating", 4.0))
        scored.append((score, item))

    scored.sort(key=lambda pair: (-pair[0], -pair[1]["rating"]))
    return [dict(item) for _, item in scored[: max(1, limit)]]


def get_weather(destination: str, dates: List[str]) -> Dict[str, Any]:
    """查询目的地指定日期的天气（模拟数据）。

    Args:
        destination: 目的地城市。
        dates: 日期字符串列表，格式 YYYY-MM-DD。

    Returns:
        形如 ``{"destination": "北京", "source": "模拟数据", "forecast": [WeatherInfo...]}``
        其中 forecast 与 dates 一一对应。
    """
    destination = (destination or "").strip()
    if isinstance(dates, str):
        dates = [dates]
    dates = list(dates or [])

    if destination in CLIMATE_BASE:
        forecast: List[Dict[str, Any]] = []
        for date_str in dates:
            try:
                day = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            seed = _stable_seed(destination, date_str, "climate")
            base = CLIMATE_BASE[destination][day.month - 1]
            offset = (seed % 5) - 2
            cond = WEATHER_CONDITIONS[seed % len(WEATHER_CONDITIONS)]
            forecast.append(
                {
                    "date": day.isoformat(),
                    "weekday": WEEKDAY_CN[day.weekday()],
                    "condition": cond["condition"],
                    "temp_min": base[0] + offset,
                    "temp_max": base[1] + offset,
                    "wind": f"{2 + seed % 4} 级",
                    "suggestion": cond["suggestion"],
                }
            )
    else:
        forecast = _weather_fallback(destination, dates)

    return {
        "destination": destination,
        "source": "模拟数据（非实时气象接口）",
        "forecast": forecast,
    }


def search_hotels(destination: str, budget_level: str = "中等", limit: int = 3) -> List[Dict[str, Any]]:
    """根据目的地与预算档位搜索酒店。

    Args:
        destination: 目的地城市。
        budget_level: 预算档位，经济 / 中等 / 豪华。
        limit: 返回条数。

    Returns:
        酒店字典列表；若该城市缺少对应档位数据，会自动降级到相邻档位，
        仍不足时用城市名确定性生成兜底数据，保证不为空。
    """
    destination = (destination or "").strip()
    level = budget_level if budget_level in ("经济", "中等", "豪华") else "中等"

    city_data = HOTELS_DB.get(destination, {})
    result: List[Dict[str, Any]] = list(city_data.get(level, []))

    # 档位降级：中等 -> 经济/豪华 -> 其它
    if not result:
        fallback_order = [lv for lv in ("中等", "经济", "豪华") if lv != level]
        for lv in fallback_order:
            if city_data.get(lv):
                result = list(city_data[lv])
                break

    # 城市完全未知：生成兜底酒店
    if not result:
        price_map = {"经济": 280, "中等": 680, "豪华": 1680}
        base_price = price_map[level]
        seed = _stable_seed(destination, level)
        result = [
            {
                "name": f"{destination}中心商务酒店",
                "price_per_night": base_price,
                "rating": round(4.2 + (seed % 5) / 10, 1),
                "location": f"{destination}市中心",
                "level": level,
                "tags": ["交通便利", "近地铁"],
                "distance_to_center": "位于市中心",
            },
            {
                "name": f"{destination}精品民宿",
                "price_per_night": int(base_price * 0.85),
                "rating": round(4.1 + (seed % 6) / 10, 1),
                "location": f"{destination}老城区",
                "level": level,
                "tags": ["本地体验", "含早"],
                "distance_to_center": "距市中心约 2.0km",
            },
            {
                "name": f"{destination}国际连锁酒店",
                "price_per_night": int(base_price * 1.25),
                "rating": round(4.4 + (seed % 5) / 10, 1),
                "location": f"{destination}新区",
                "level": level,
                "tags": ["健身房", "含早"],
                "distance_to_center": "距市中心约 4.0km",
            },
        ]

    # 按价格升序，保证"经济优先"的直觉
    result.sort(key=lambda item: item.get("price_per_night", 0))
    return [dict(item) for item in result[: max(1, limit)]]


# ---------------------------------------------------------------------------
# 五、工具 JSON Schema（供 DeepSeek function calling 使用）
# ---------------------------------------------------------------------------
def search_attractions_tool_spec() -> Dict[str, Any]:
    """search_attractions 的 function calling 描述。"""
    return {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "根据目的地城市和用户偏好标签搜索景点，返回景点名称、简介、建议游览时长与门票价格。",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "目的地城市，例如：北京"},
                    "preference": {
                        "type": "string",
                        "description": "偏好关键词，可多个用逗号分隔，例如：历史文化,美食",
                    },
                    "limit": {"type": "integer", "description": "最多返回的景点数量，默认 8"},
                },
                "required": ["destination"],
            },
        },
    }


def get_weather_tool_spec() -> Dict[str, Any]:
    """get_weather 的 function calling 描述。"""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询目的地指定日期的天气情况，返回天气状况、温度区间与出行建议。",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "目的地城市，例如：北京"},
                    "dates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "日期列表，格式 YYYY-MM-DD，例如：['2025-05-01','2025-05-02']",
                    },
                },
                "required": ["destination", "dates"],
            },
        },
    }


def search_hotels_tool_spec() -> Dict[str, Any]:
    """search_hotels 的 function calling 描述。"""
    return {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "根据目的地城市和预算档位搜索酒店，返回酒店名称、每晚价格、评分与位置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "目的地城市，例如：北京"},
                    "budget_level": {
                        "type": "string",
                        "enum": ["经济", "中等", "豪华"],
                        "description": "预算档位",
                    },
                    "limit": {"type": "integer", "description": "返回条数，默认 3"},
                },
                "required": ["destination", "budget_level"],
            },
        },
    }


# 工具名 -> 可调用函数 的注册表，供 Agent 分发 function call
TOOL_REGISTRY: Dict[str, Any] = {
    "search_attractions": search_attractions,
    "get_weather": get_weather,
    "search_hotels": search_hotels,
}

# 工具名 -> JSON Schema 的注册表
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "search_attractions": search_attractions_tool_spec(),
    "get_weather": get_weather_tool_spec(),
    "search_hotels": search_hotels_tool_spec(),
}


def get_supported_destinations() -> List[str]:
    """返回内置模拟数据覆盖的所有目的地城市。"""
    return sorted(ATTRACTIONS_DB.keys())


# ---------------------------------------------------------------------------
# 六、Pydantic 转换辅助函数
# ---------------------------------------------------------------------------
def to_attraction_models(raw_items: List[Dict[str, Any]]) -> List[Attraction]:
    """把原始字典转换成 Attraction 模型列表。"""
    return [Attraction(**item) for item in raw_items]


def to_hotel_models(raw_items: List[Dict[str, Any]]) -> List[Hotel]:
    """把原始字典转换成 Hotel 模型列表。"""
    return [Hotel(**item) for item in raw_items]


def to_weather_models(raw_forecast: List[Dict[str, Any]]) -> List[WeatherInfo]:
    """把原始字典转换成 WeatherInfo 模型列表。"""
    return [WeatherInfo(**item) for item in raw_forecast]


def weather_text(weather: WeatherInfo) -> str:
    """把天气模型压缩成一行可读文本，供提示词使用。"""
    return (
        f"{weather.date}（{weather.weekday}）{weather.condition} "
        f"{weather.temp_range} {weather.wind}"
    )
