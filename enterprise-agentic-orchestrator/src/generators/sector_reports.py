"""Sector analysis document generator for RAG ingestion.

Produces 10 UK-specific sector analysis documents (one per SectorType)
as structured dictionaries with markdown content fields suitable for
Weaviate vector storage and semantic search.
"""

from datetime import datetime
from uuid import uuid4

from src.models.loan import SectorType


# Sector-specific data for each of the 10 industries
_SECTOR_DATA: dict[str, dict] = {
    SectorType.CONSTRUCTION.value: {
        "outlook": "cautious",
        "risk_level": "high",
        "key_metrics": {
            "sector_growth_rate": 0.02,
            "default_rate": 0.065,
            "average_profit_margin": 0.04,
            "employment_trend": "stable",
        },
        "key_risks": [
            "Rising material costs driven by global supply chain disruptions and post-Brexit tariff adjustments",
            "Skilled labour shortages exacerbated by tighter immigration controls and an ageing workforce",
            "Planning permission delays increasing project timelines and working capital requirements",
            "Interest rate sensitivity on large-scale developments funded by variable-rate facilities",
            "Regulatory tightening around building safety standards following the Building Safety Act 2022",
        ],
        "opportunities": [
            "Government infrastructure spending commitments under the National Infrastructure Strategy",
            "Retrofit and energy efficiency programmes driven by net-zero targets for 2050",
            "Growing demand for data centre and logistics facility construction",
        ],
        "content": (
            "## UK Construction Sector Analysis 2024\n\n"
            "The UK construction sector continues to navigate a complex operating environment "
            "characterised by elevated input costs, persistent labour shortages, and evolving "
            "regulatory requirements. Total construction output reached GBP 171 billion in 2023, "
            "representing a modest 2% year-on-year increase. However, the sector's default rate "
            "of 6.5% remains among the highest across UK commercial lending portfolios, reflecting "
            "the inherent cyclicality and project-execution risks endemic to the industry.\n\n"
            "The Building Safety Act 2022 has introduced significant compliance obligations for "
            "developers, particularly in the residential high-rise segment. Firms lacking the "
            "capital reserves to absorb remediation costs or adapt to new safety standards face "
            "elevated insolvency risk. Lenders should assess counterparty exposure to legacy "
            "building stock and the adequacy of professional indemnity cover when evaluating "
            "construction borrowers.\n\n"
            "Material cost inflation, while moderating from 2022 peaks, continues to compress "
            "margins. Timber, steel, and concrete prices remain 15-25% above pre-pandemic levels. "
            "Fixed-price contracts entered during lower-cost periods pose particular risk, with "
            "several mid-tier contractors reporting margin erosion of 200-300 basis points. "
            "Working capital facilities should be stress-tested against further cost escalation "
            "scenarios.\n\n"
            "On the positive side, government capital expenditure programmes -- including HS2 "
            "phases, hospital building, and school refurbishment -- provide a pipeline of publicly "
            "funded work that supports revenue visibility for Tier 1 and Tier 2 contractors. "
            "The retrofit economy, driven by Energy Performance Certificate requirements and "
            "net-zero commitments, represents a structural growth opportunity estimated at "
            "GBP 9.2 billion annually by 2030."
        ),
    },
    SectorType.HOSPITALITY.value: {
        "outlook": "neutral",
        "risk_level": "medium",
        "key_metrics": {
            "sector_growth_rate": 0.04,
            "default_rate": 0.055,
            "average_profit_margin": 0.06,
            "employment_trend": "growing",
        },
        "key_risks": [
            "Consumer discretionary spending pressure from cost-of-living crisis and elevated inflation",
            "Staff recruitment and retention challenges with sector vacancy rates above 8%",
            "Energy cost volatility disproportionately affecting high-consumption premises",
            "Seasonal cash flow variability creating working capital strain in Q1 and Q4",
        ],
        "opportunities": [
            "Domestic tourism growth as staycation trends persist post-pandemic",
            "Premiumisation trend with consumers trading up on fewer occasions",
            "Technology adoption in booking platforms and operational efficiency",
        ],
        "content": (
            "## UK Hospitality Sector Analysis 2024\n\n"
            "The UK hospitality sector has demonstrated resilience in its post-pandemic recovery, "
            "with total revenue approaching GBP 130 billion in 2023 and year-on-year growth of "
            "4%. However, the sector remains under considerable pressure from the cost-of-living "
            "crisis, which is constraining consumer discretionary spending. The default rate of "
            "5.5% reflects ongoing viability challenges, particularly among independent operators "
            "without the scale efficiencies of larger chains.\n\n"
            "Labour market dynamics remain the sector's most acute operational challenge. Vacancy "
            "rates in hospitality exceed 8%, well above the UK average of 3.5%. The post-Brexit "
            "reduction in EU migrant workers has structurally tightened the available labour pool, "
            "driving wage inflation of 7-9% across front-of-house and kitchen roles. Operators "
            "unable to pass these costs to consumers face sustained margin compression.\n\n"
            "Energy costs, while declining from 2022 peaks, continue to represent a disproportionate "
            "burden for hospitality businesses. Hotels and restaurants typically consume 3-5x more "
            "energy per square metre than office premises. The expiry of government energy support "
            "schemes has left many operators exposed to market-rate contracts, with annual energy "
            "bills for mid-sized hotels ranging from GBP 150,000 to GBP 400,000.\n\n"
            "The domestic tourism market presents a structural tailwind. UK staycation bookings "
            "remain 22% above 2019 levels, supported by the weak pound discouraging outbound travel "
            "and a growing consumer preference for experiential holidays. Lenders should favour "
            "borrowers with exposure to leisure and tourism sub-segments and demonstrated ability "
            "to manage seasonal cash flow variability through revolving credit facilities."
        ),
    },
    SectorType.RETAIL.value: {
        "outlook": "negative",
        "risk_level": "high",
        "key_metrics": {
            "sector_growth_rate": -0.01,
            "default_rate": 0.072,
            "average_profit_margin": 0.03,
            "employment_trend": "declining",
        },
        "key_risks": [
            "Structural decline of physical retail accelerated by e-commerce penetration exceeding 30%",
            "High street vacancy rates at 14% nationally with regional centres above 20%",
            "Thin operating margins leaving minimal buffer against cost shocks",
            "Consumer confidence indices at historically low levels constraining discretionary spend",
            "Business rates burden disproportionately penalising bricks-and-mortar operators",
        ],
        "opportunities": [
            "Omnichannel integration creating defensible competitive positions for adapted retailers",
            "Value retail segment outperforming as consumers trade down during cost-of-living pressures",
        ],
        "content": (
            "## UK Retail Sector Analysis 2024\n\n"
            "The UK retail sector continues its structural transformation, with total sales declining "
            "1% in real terms during 2023. The sector's default rate of 7.2% is the highest across "
            "our commercial lending portfolio, driven by the ongoing migration of consumer spending "
            "to online channels and the erosion of margins through cost inflation. E-commerce now "
            "accounts for over 30% of total retail sales, up from 19% pre-pandemic, and this shift "
            "shows no signs of reversing.\n\n"
            "High street vacancy rates present a stark indicator of sector distress. The national "
            "average of 14% masks significant regional variation, with northern and Midlands town "
            "centres experiencing vacancy rates above 20%. The business rates system continues to "
            "disadvantage physical retail, with reform proposals progressing slowly through "
            "government consultation. CVAs and administrations among mid-market fashion and "
            "homeware retailers remain elevated.\n\n"
            "Operating margins in non-food retail have compressed to approximately 3%, leaving "
            "minimal buffer against further cost shocks. Rising minimum wage obligations, packaging "
            "levy implementation, and supply chain costs collectively add 150-200 basis points of "
            "cost pressure annually. Retailers without differentiated product propositions or "
            "significant online revenue face existential viability questions.\n\n"
            "Within this challenging landscape, the value retail segment demonstrates relative "
            "resilience. Discount retailers, charity shops, and off-price formats continue to "
            "capture market share as consumers trade down. Lenders should apply enhanced scrutiny "
            "to fashion, homeware, and general merchandise retailers while recognising that food "
            "retail and essential goods operators maintain defensible market positions."
        ),
    },
    SectorType.TECHNOLOGY.value: {
        "outlook": "positive",
        "risk_level": "low",
        "key_metrics": {
            "sector_growth_rate": 0.08,
            "default_rate": 0.025,
            "average_profit_margin": 0.15,
            "employment_trend": "growing",
        },
        "key_risks": [
            "Talent competition driving wage inflation of 10-15% for specialist AI and cloud roles",
            "Regulatory uncertainty around AI governance and data protection post-GDPR reform",
            "Customer concentration risk in B2B SaaS businesses with enterprise-dependent revenue",
        ],
        "opportunities": [
            "AI and machine learning adoption creating demand across all verticals",
            "UK government digital transformation programmes valued at GBP 2.3 billion annually",
            "Cybersecurity spending growth of 12% year-on-year driven by threat landscape evolution",
        ],
        "content": (
            "## UK Technology Sector Analysis 2024\n\n"
            "The UK technology sector remains the strongest performer in our commercial lending "
            "portfolio, with 8% revenue growth in 2023 and a sector default rate of just 2.5%. "
            "The UK's position as Europe's leading technology hub is underpinned by deep talent "
            "pools, favourable regulatory frameworks, and strong venture capital ecosystems in "
            "London, Cambridge, and the Northern tech corridor. Total sector revenue exceeded "
            "GBP 200 billion, contributing approximately 10% of UK GDP.\n\n"
            "The artificial intelligence boom has created significant opportunities for UK technology "
            "firms, with AI-related revenue growing 35% year-on-year. The UK AI Safety Institute, "
            "established following the Bletchley Park summit, positions the UK as a credible "
            "regulatory leader, which in turn attracts international AI investment. Enterprise "
            "adoption of generative AI tools is driving demand for consulting, integration, and "
            "infrastructure services across the technology value chain.\n\n"
            "Cybersecurity represents another structural growth driver, with UK organisations "
            "increasing security budgets by 12% annually in response to escalating threat "
            "sophistication. The sector benefits from mandatory breach reporting under the UK GDPR "
            "and Network and Information Systems Regulations, which compel ongoing investment in "
            "security infrastructure and services.\n\n"
            "Key lending risks in technology centre on talent-driven cost inflation and customer "
            "concentration. Specialist engineers in AI, cloud infrastructure, and cybersecurity "
            "command salary premiums of 30-50% over general software development roles. B2B SaaS "
            "businesses should be assessed for revenue concentration, with exposure to any single "
            "customer exceeding 20% of annual recurring revenue flagged for enhanced monitoring."
        ),
    },
    SectorType.MANUFACTURING.value: {
        "outlook": "cautious",
        "risk_level": "medium",
        "key_metrics": {
            "sector_growth_rate": 0.01,
            "default_rate": 0.048,
            "average_profit_margin": 0.06,
            "employment_trend": "stable",
        },
        "key_risks": [
            "Global supply chain fragmentation increasing input costs and lead times",
            "Sterling volatility affecting export competitiveness and import costs",
            "Energy-intensive processes exposed to wholesale electricity and gas price fluctuations",
            "Post-Brexit customs friction adding 4-7% to EU trade administrative costs",
        ],
        "opportunities": [
            "Reshoring trend as firms prioritise supply chain resilience over pure cost optimisation",
            "Advanced manufacturing and Industry 4.0 investment improving productivity",
            "Defence and aerospace spending increases supporting UK manufacturing capabilities",
        ],
        "content": (
            "## UK Manufacturing Sector Analysis 2024\n\n"
            "UK manufacturing output grew by a modest 1% in 2023, reflecting a sector in transition "
            "between post-pandemic recovery and structural adaptation to new trade realities. The "
            "sector's default rate of 4.8% sits in the medium-risk band, with significant "
            "dispersion between sub-segments. Aerospace, defence, and pharmaceutical manufacturing "
            "demonstrate strong fundamentals, while commodity-exposed and energy-intensive "
            "manufacturers face more challenging conditions.\n\n"
            "Post-Brexit customs and regulatory arrangements continue to impose friction costs on "
            "UK manufacturers with EU supply chains. Industry surveys indicate that customs "
            "documentation, rules of origin compliance, and border delays add 4-7% to the cost "
            "of EU trade for mid-sized manufacturers. Firms that have diversified supply chains "
            "towards non-EU partners or established EU-based subsidiaries demonstrate superior "
            "operational resilience.\n\n"
            "Energy costs remain a critical variable for UK manufacturing competitiveness. "
            "Industrial electricity prices in the UK are 30-50% higher than those in France and "
            "Germany, placing energy-intensive sectors -- ceramics, glass, steel, chemicals -- at "
            "a structural disadvantage. The government's Energy Intensive Industries compensation "
            "scheme provides partial relief, but eligibility criteria exclude many mid-tier "
            "manufacturers.\n\n"
            "The reshoring and nearshoring trend represents a countervailing positive force. "
            "Supply chain disruptions during 2020-2022 have prompted a reassessment of "
            "offshoring strategies, with 23% of UK manufacturers reporting active reshoring "
            "plans. Advanced manufacturing technologies -- robotics, additive manufacturing, "
            "digital twins -- are enabling UK firms to compete on quality and flexibility "
            "rather than labour cost alone."
        ),
    },
    SectorType.HEALTHCARE.value: {
        "outlook": "positive",
        "risk_level": "low",
        "key_metrics": {
            "sector_growth_rate": 0.06,
            "default_rate": 0.018,
            "average_profit_margin": 0.10,
            "employment_trend": "growing",
        },
        "key_risks": [
            "Regulatory compliance costs increasing with CQC inspection framework changes",
            "Workforce shortages with NHS vacancy rates creating competitive pressure on private providers",
            "Government policy risk around private healthcare funding and NHS reform",
        ],
        "opportunities": [
            "NHS waiting list backlog driving demand for private healthcare services",
            "Digital health and telemedicine adoption accelerating post-pandemic",
            "Ageing population creating structural demand growth for care services",
        ],
        "content": (
            "## UK Healthcare Sector Analysis 2024\n\n"
            "The UK healthcare sector presents one of the strongest credit profiles in our "
            "lending portfolio, with 6% revenue growth, a default rate of just 1.8%, and "
            "average operating margins of 10%. The sector benefits from non-discretionary "
            "demand drivers, demographic tailwinds from an ageing population, and a structural "
            "capacity gap between NHS provision and patient needs that sustains demand for "
            "private healthcare services.\n\n"
            "The NHS waiting list, which peaked at 7.8 million patients in 2023, continues to "
            "drive volumes into the private healthcare sector. Self-pay and private medical "
            "insurance funded treatments have grown 18% year-on-year, with particular strength "
            "in diagnostics, orthopaedics, and mental health services. Private hospital operators "
            "report occupancy rates exceeding 85%, well above breakeven thresholds.\n\n"
            "Digital health and telemedicine represent a rapidly growing sub-segment, with UK "
            "digital health revenues exceeding GBP 4 billion in 2023. Remote consultation "
            "platforms, AI-assisted diagnostics, and digital therapeutics are attracting "
            "significant venture capital investment. The NHS App ecosystem and NHSX digital "
            "transformation programme create interoperability requirements that favour UK-based "
            "health technology firms with existing NHS data-sharing agreements.\n\n"
            "Lending risks in healthcare are primarily regulatory in nature. The Care Quality "
            "Commission's evolving inspection framework and the Health and Care Act 2022's "
            "provisions around provider licensing create compliance obligations that require "
            "ongoing investment. Borrowers should demonstrate robust clinical governance "
            "frameworks and adequate professional indemnity insurance. Workforce dependency "
            "on international recruitment should be assessed against Home Office visa policy "
            "changes."
        ),
    },
    SectorType.LOGISTICS.value: {
        "outlook": "neutral",
        "risk_level": "medium",
        "key_metrics": {
            "sector_growth_rate": 0.03,
            "default_rate": 0.042,
            "average_profit_margin": 0.05,
            "employment_trend": "stable",
        },
        "key_risks": [
            "Fuel cost volatility directly impacting operating margins for haulage operators",
            "Driver shortage persisting with an estimated 40,000 HGV driver deficit",
            "E-commerce delivery margin pressure from consumer expectations of free shipping",
            "Regulatory costs from clean air zones and vehicle emission standards",
        ],
        "opportunities": [
            "E-commerce fulfilment growth sustaining demand for last-mile delivery infrastructure",
            "Warehouse automation and autonomous vehicle technology reducing operating costs",
            "Cold chain logistics expansion driven by online grocery and pharmaceutical distribution",
        ],
        "content": (
            "## UK Logistics Sector Analysis 2024\n\n"
            "The UK logistics sector generated revenues of approximately GBP 124 billion in 2023, "
            "growing 3% year-on-year as e-commerce fulfilment volumes continued to expand. The "
            "sector's default rate of 4.2% reflects the operational challenges of a capital-intensive, "
            "margin-sensitive industry where fuel costs, labour availability, and vehicle utilisation "
            "rates are critical determinants of financial performance.\n\n"
            "The HGV driver shortage remains the sector's most persistent structural challenge. "
            "Despite a partial recovery from the acute crisis of 2021, an estimated 40,000 driver "
            "deficit persists. Driver wages have increased by 15-20% since 2020, compressing margins "
            "for haulage operators. The introduction of the Driver Certificate of Professional "
            "Competence and enhanced licence requirements has slowed the pipeline of new drivers "
            "entering the workforce.\n\n"
            "Last-mile delivery economics continue to challenge logistics operators. Consumer "
            "expectations of free or low-cost delivery, driven by marketplace platforms, create "
            "persistent margin pressure. The average cost of a last-mile delivery in the UK is "
            "GBP 5.50-7.00, while consumer willingness to pay averages GBP 3.50. This gap is "
            "funded through retailer subsidies, but rising delivery volumes and urban congestion "
            "are straining the model.\n\n"
            "Clean air zones in major UK cities present both a cost and an opportunity. Compliance "
            "with Euro VI emission standards and London's Ultra Low Emission Zone requires "
            "fleet investment, with electric HGV costs approximately 2.5x diesel equivalents. "
            "However, operators that invest early in fleet electrification gain competitive "
            "advantage through lower operating costs and preferential access to urban delivery zones."
        ),
    },
    SectorType.PROFESSIONAL_SERVICES.value: {
        "outlook": "positive",
        "risk_level": "low",
        "key_metrics": {
            "sector_growth_rate": 0.05,
            "default_rate": 0.022,
            "average_profit_margin": 0.18,
            "employment_trend": "growing",
        },
        "key_risks": [
            "Partner departure risk creating revenue concentration vulnerability in smaller firms",
            "Fee rate pressure from corporate procurement teams and panel arrangements",
            "Professional indemnity insurance cost increases of 15-20% annually",
        ],
        "opportunities": [
            "Regulatory complexity driving demand for compliance, legal, and advisory services",
            "Cross-border M&A advisory growth as UK firms attract international investment",
            "AI-augmented service delivery improving margins in legal and accounting practices",
        ],
        "content": (
            "## UK Professional Services Sector Analysis 2024\n\n"
            "The UK professional services sector continues to demonstrate strong credit "
            "fundamentals, with 5% revenue growth, a default rate of just 2.2%, and sector-leading "
            "average margins of 18%. The sector -- encompassing legal, accounting, consulting, and "
            "specialist advisory firms -- benefits from the UK's position as a global centre for "
            "financial and professional services, with London alone generating over GBP 80 billion "
            "in professional services revenue.\n\n"
            "Regulatory complexity is a structural demand driver for professional services. The "
            "post-Brexit regulatory divergence, evolving ESG disclosure requirements, and "
            "increasing anti-money laundering obligations generate sustained advisory demand. "
            "The Big Four accounting firms reported combined UK revenues exceeding GBP 16 billion "
            "in 2023, with audit and assurance practices growing 8% driven by enhanced reporting "
            "requirements under the proposed Audit Reform Bill.\n\n"
            "The legal sector, valued at GBP 44 billion, demonstrates particular resilience. "
            "English law remains the governing law of choice for international commercial "
            "contracts, supporting London's magic circle and silver circle firms. Litigation "
            "volumes have increased post-pandemic, with insolvency-related and contractual "
            "dispute work providing counter-cyclical revenue streams.\n\n"
            "Key lending considerations for professional services firms centre on partner "
            "economics and human capital dependency. Revenue per partner, profit per equity "
            "point, and client concentration metrics should be assessed alongside traditional "
            "financial ratios. Firms with diversified practice areas, institutional client "
            "relationships, and strong associate-to-partner pipelines present lower credit risk "
            "than specialist boutiques dependent on individual rainmaker partners."
        ),
    },
    SectorType.AGRICULTURE.value: {
        "outlook": "cautious",
        "risk_level": "high",
        "key_metrics": {
            "sector_growth_rate": 0.00,
            "default_rate": 0.058,
            "average_profit_margin": 0.04,
            "employment_trend": "declining",
        },
        "key_risks": [
            "Post-Brexit subsidy transition from BPS to ELMS creating income uncertainty for farms",
            "Climate volatility with increasing frequency of drought and flood events",
            "Input cost inflation for fertiliser, feed, and fuel exceeding commodity price recovery",
            "Generational succession challenges with average farmer age exceeding 59 years",
            "Trade policy risk from potential import deals undermining domestic price floors",
        ],
        "opportunities": [
            "Environmental Land Management Scheme payments for nature recovery and carbon sequestration",
            "Agri-tech adoption improving yields and reducing input costs",
        ],
        "content": (
            "## UK Agriculture Sector Analysis 2024\n\n"
            "The UK agriculture sector faces a period of unprecedented structural adjustment, "
            "with zero real growth in 2023 and a default rate of 5.8% that places it firmly in "
            "the high-risk category. The transition from the EU Common Agricultural Policy's "
            "Basic Payment Scheme to the domestic Environmental Land Management Scheme (ELMS) "
            "represents the most significant change to agricultural economics in a generation. "
            "Direct payments, which historically contributed 50-80% of farm net income, are being "
            "phased down through 2027.\n\n"
            "The ELMS transition creates acute cash flow risk for arable and livestock farms that "
            "have historically relied on area-based subsidies. While the new scheme offers payments "
            "for environmental stewardship, nature recovery, and landscape management, uptake has "
            "been slower than projected. Defra estimates that only 38% of eligible farms have "
            "enrolled in the Sustainable Farming Incentive, creating an income gap for the "
            "majority.\n\n"
            "Climate volatility is an intensifying risk factor. The 2022 drought cost UK "
            "agriculture an estimated GBP 1.4 billion in lost production, while winter flooding "
            "in 2023-24 rendered significant areas of productive farmland temporarily unusable. "
            "Crop insurance penetration remains low in the UK compared to US and Continental "
            "European markets, leaving many farms unhedged against weather events.\n\n"
            "Lenders to the agricultural sector should prioritise borrowers demonstrating active "
            "diversification -- farm shops, tourism, renewable energy generation -- alongside "
            "core agricultural production. Land-secured lending benefits from the long-term "
            "appreciation trend in UK agricultural land values, which have increased 150% over "
            "the past decade, but forced-sale scenarios during cash flow distress may not realise "
            "full market values."
        ),
    },
    SectorType.ENERGY.value: {
        "outlook": "neutral",
        "risk_level": "medium",
        "key_metrics": {
            "sector_growth_rate": 0.07,
            "default_rate": 0.035,
            "average_profit_margin": 0.09,
            "employment_trend": "growing",
        },
        "key_risks": [
            "Commodity price volatility creating revenue unpredictability for producers",
            "Regulatory and planning risk for renewable energy project development",
            "Grid connection delays of 10-15 years constraining renewable capacity deployment",
            "Stranded asset risk for fossil fuel infrastructure under net-zero transition",
        ],
        "opportunities": [
            "Offshore wind capacity expansion with UK targeting 50GW by 2030",
            "Green hydrogen and battery storage as emerging investable sub-sectors",
            "Energy efficiency and heat pump installation driven by Future Homes Standard",
        ],
        "content": (
            "## UK Energy Sector Analysis 2024\n\n"
            "The UK energy sector is experiencing a period of rapid transformation, with 7% "
            "revenue growth in 2023 driven primarily by the renewable energy transition. The "
            "sector's default rate of 3.5% reflects a balance between the stability of regulated "
            "utility businesses and the development risk inherent in capital-intensive renewable "
            "energy projects. Total UK energy sector investment exceeded GBP 22 billion in 2023, "
            "with renewables accounting for over 60% of new capacity additions.\n\n"
            "Offshore wind remains the flagship of UK energy policy, with the government's target "
            "of 50GW installed capacity by 2030 requiring an investment of approximately GBP 50 "
            "billion. The Contracts for Difference (CfD) auction mechanism provides revenue "
            "certainty for developers, but recent allocation rounds have seen strike prices fall "
            "below project economics, leading to under-subscription. Lenders should assess CfD "
            "contract terms, construction risk mitigation, and counterparty exposure to LCCC "
            "payments.\n\n"
            "Grid infrastructure is the binding constraint on UK energy transition. National Grid "
            "ESO reports a connection queue exceeding 400GW, with average connection timelines of "
            "10-15 years for new generation projects. This bottleneck creates stranding risk for "
            "projects with planning consent but without grid connection agreements. Battery "
            "storage, which can deploy more rapidly and provide grid balancing services, represents "
            "an attractive sub-sector with strong risk-adjusted returns.\n\n"
            "The transition creates stranded asset risk for fossil fuel infrastructure. Gas-fired "
            "power stations, while currently essential for grid stability, face declining capacity "
            "factors and uncertain long-term economics. Oil and gas exploration companies operating "
            "UK Continental Shelf assets confront the Energy Profits Levy and declining reservoir "
            "productivity. Lenders should apply enhanced climate transition risk assessment to "
            "all fossil fuel exposures and establish clear portfolio decarbonisation pathways."
        ),
    },
}


def generate_sector_reports() -> list[dict]:
    """Generate 10 sector analysis documents for RAG ingestion.

    Returns a list of structured dictionaries, one per SectorType,
    containing metadata fields for filtering and markdown content
    for semantic search in Weaviate.
    """
    reports: list[dict] = []

    for sector in SectorType:
        data = _SECTOR_DATA[sector.value]
        report = {
            "document_id": str(uuid4()),
            "document_type": "sector_analysis",
            "sector": sector.value,
            "title": f"{sector.value.replace('_', ' ').title()} Sector Analysis 2024",
            "outlook": data["outlook"],
            "risk_level": data["risk_level"],
            "key_metrics": data["key_metrics"],
            "key_risks": data["key_risks"],
            "opportunities": data["opportunities"],
            "content": data["content"],
            "generated_at": datetime.utcnow().isoformat(),
        }
        reports.append(report)

    return reports
