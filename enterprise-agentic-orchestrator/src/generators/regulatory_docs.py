"""Regulatory policy document generator for RAG ingestion.

Produces 4 UK regulatory policy documents covering FCA Consumer Duty,
fair lending guidelines, risk appetite framework, and concentration limits.
Structured as dictionaries with markdown content fields suitable for
Weaviate vector storage and semantic search.
"""

from datetime import datetime
from uuid import uuid4


_REGULATORY_POLICIES: list[dict] = [
    {
        "policy_area": "consumer_duty",
        "title": "FCA Consumer Duty: Commercial Lending Compliance Framework",
        "regulation_reference": "FCA PS22/9, FG22/5",
        "effective_date": "2023-07-31",
        "key_requirements": [
            "Act to deliver good outcomes for retail and SME customers across all products and services",
            "Ensure price and value assessments demonstrate fair relationship between cost and benefit",
            "Provide clear and timely communications enabling informed decision-making by borrowers",
            "Deliver appropriate customer support throughout the product lifecycle including forbearance",
            "Maintain ongoing monitoring of customer outcomes with board-level accountability",
            "Apply the cross-cutting rules: act in good faith, avoid foreseeable harm, enable customers to pursue financial objectives",
        ],
        "compliance_checks": [
            "Annual price and value assessment completed for all lending products",
            "Customer communications reviewed against FCA clarity standards (FG22/5 Annex 2)",
            "Vulnerable customer identification and treatment processes documented and tested",
            "Outcome monitoring MI produced quarterly with escalation thresholds defined",
            "Product governance framework includes Consumer Duty impact assessment for new products",
        ],
        "content": (
            "## FCA Consumer Duty: Commercial Lending Compliance Framework\n\n"
            "### Overview and Regulatory Context\n\n"
            "The FCA Consumer Duty, implemented through PS22/9 and supplemented by FG22/5, "
            "represents the most significant conduct regulation reform in UK financial services "
            "since the Retail Distribution Review. Effective from 31 July 2023 for new and "
            "existing products, the Duty establishes a higher standard of care that firms must "
            "provide to customers. While primarily targeting retail financial services, the Duty's "
            "scope extends to SME lending where borrowers fall within the FCA's regulatory "
            "perimeter, including regulated credit agreements and insurance distribution.\n\n"
            "### Consumer Outcomes Framework\n\n"
            "The Duty is structured around four outcomes that firms must deliver: products and "
            "services (appropriate design and distribution), price and value (fair relationship "
            "between total cost and benefit), consumer understanding (clear communications "
            "enabling informed decisions), and consumer support (accessible and responsive "
            "service throughout the customer journey). For commercial lending, these outcomes "
            "translate into specific operational requirements.\n\n"
            "**Products and services:** Lending products must be designed with a clearly defined "
            "target market and distribution strategy. Product governance arrangements must assess "
            "whether features, terms, and conditions are consistent with the needs of the target "
            "market. This includes consideration of interest rate structures, fee arrangements, "
            "covenant packages, and security requirements. Products should not be distributed "
            "outside the identified target market without enhanced suitability assessment.\n\n"
            "**Price and value:** Firms must conduct fair value assessments that consider the "
            "total cost to the borrower, including arrangement fees, commitment fees, early "
            "repayment charges, and ongoing service charges. The assessment must demonstrate "
            "that the price paid is reasonable relative to the benefits received, benchmarked "
            "against comparable market offerings. Cross-subsidisation between customer groups "
            "must not result in identifiable groups paying significantly more than the fair value "
            "of the product they receive.\n\n"
            "**Consumer understanding:** All customer-facing communications -- from marketing "
            "materials through to annual review letters -- must be clear, fair, and not "
            "misleading. The FCA expects firms to test comprehension with representative "
            "samples of their target market. Key information, including total cost of credit, "
            "material risks, and borrower obligations, must be presented prominently and in "
            "plain language. Technical financial terminology should be explained or avoided.\n\n"
            "### Implementation Requirements\n\n"
            "Senior Management Function holders bear personal accountability for Consumer Duty "
            "compliance within their areas of responsibility. The board must review an annual "
            "Consumer Duty compliance report that assesses outcomes delivered, identifies areas "
            "of concern, and sets remediation priorities. Management information must enable "
            "real-time monitoring of customer outcomes, with defined thresholds for escalation."
        ),
    },
    {
        "policy_area": "fair_lending",
        "title": "Fair Lending and Non-Discrimination Policy: Credit Decision Framework",
        "regulation_reference": "Equality Act 2010, FCA PRIN 6",
        "effective_date": "2010-10-01",
        "key_requirements": [
            "Credit decisions must not discriminate on the basis of any protected characteristic under the Equality Act 2010",
            "Automated credit scoring models must be regularly tested for disparate impact across demographic groups",
            "Pricing decisions must be justifiable on risk-based grounds with no systematic disadvantage to protected groups",
            "Reasonable adjustments must be made for disabled applicants throughout the application and servicing process",
            "Staff training on unconscious bias and fair lending principles must be completed annually",
            "Declined applications must include clear reasons enabling the applicant to understand and challenge the decision",
        ],
        "compliance_checks": [
            "Bi-annual disparate impact analysis conducted on approval rates across protected characteristics",
            "Credit scoring model validation includes fairness metrics (demographic parity, equalised odds)",
            "Pricing exception reports reviewed monthly for patterns indicating discriminatory outcomes",
            "Complaints data analysed quarterly for fair lending themes with root cause investigation",
            "Mystery shopping programme includes fair lending scenarios across branch and digital channels",
        ],
        "content": (
            "## Fair Lending and Non-Discrimination Policy\n\n"
            "### Legal Framework\n\n"
            "This policy establishes the firm's compliance framework for fair lending obligations "
            "arising under the Equality Act 2010, the FCA's Principles for Businesses (PRIN 6: "
            "Treating Customers Fairly), and associated regulatory guidance. The Equality Act "
            "prohibits direct and indirect discrimination in the provision of services, including "
            "financial services, on the basis of nine protected characteristics: age, disability, "
            "gender reassignment, marriage and civil partnership, pregnancy and maternity, race, "
            "religion or belief, sex, and sexual orientation.\n\n"
            "### Application to Credit Decisions\n\n"
            "All credit decisions -- including initial approval, pricing, terms, limit setting, "
            "and ongoing account management -- must be made on the basis of legitimate, "
            "risk-relevant criteria. The use of protected characteristics, or proxies for "
            "protected characteristics, as inputs to credit scoring models or manual underwriting "
            "assessments is prohibited. Where geographical data (postcode, region) is used as a "
            "risk factor, the firm must demonstrate that its predictive value is not primarily "
            "derived from correlation with ethnic composition or socioeconomic deprivation.\n\n"
            "### Automated Decision-Making and AI Fairness\n\n"
            "The increasing use of machine learning models in credit decisioning introduces "
            "specific fair lending risks. Models trained on historical lending data may "
            "perpetuate or amplify existing biases present in that data. The firm requires "
            "that all automated credit decision models undergo fairness testing prior to "
            "deployment and at bi-annual intervals thereafter. Fairness metrics must include "
            "demographic parity (approval rates), equalised odds (true positive and false "
            "positive rates), and calibration (predicted probability accuracy) across protected "
            "groups.\n\n"
            "Where disparate impact is identified -- defined as an adverse outcome ratio exceeding "
            "0.80 (the four-fifths rule) -- the model owner must conduct a business necessity "
            "analysis demonstrating that the disparate impact is attributable to legitimate risk "
            "factors and cannot be reduced through less discriminatory alternative model "
            "specifications. The FCA's guidance on AI transparency (DP5/22) reinforces the "
            "expectation that firms can explain how automated decisions are reached and "
            "demonstrate that outcomes are fair.\n\n"
            "### Monitoring and Governance\n\n"
            "The Fair Lending Officer, reporting to the Chief Risk Officer, maintains oversight "
            "of the firm's fair lending programme. Quarterly fair lending dashboards are produced "
            "covering approval rates, pricing distributions, and complaint volumes segmented by "
            "available demographic data. Annual fair lending risk assessments identify emerging "
            "risks and inform the compliance testing programme. All fair lending incidents are "
            "reported to the board Risk Committee within 5 business days."
        ),
    },
    {
        "policy_area": "risk_appetite",
        "title": "Credit Risk Appetite Framework: Commercial Lending Portfolio",
        "regulation_reference": "PRA SS1/23, Basel III",
        "effective_date": "2024-01-01",
        "key_requirements": [
            "Board-approved risk appetite statement reviewed and ratified annually",
            "Quantitative risk appetite metrics defined with green/amber/red thresholds for all material risk types",
            "Portfolio concentration limits established for sector, geography, single-name, and product type",
            "Stress testing results integrated into risk appetite calibration and limit-setting processes",
            "Risk appetite cascaded to business units through delegated authorities and mandate limits",
            "Monthly risk appetite utilisation reporting to Executive Risk Committee with breach escalation protocols",
        ],
        "compliance_checks": [
            "Board risk appetite statement signed off within last 12 months",
            "All quantitative RAMs (Risk Appetite Metrics) within approved thresholds or formally breached with remediation plan",
            "Stress test results from latest ICAAP/ILAAP incorporated into risk appetite calibration",
            "Delegated authority framework aligned to current risk appetite with no stale mandates",
            "Risk appetite MI produced monthly with trend analysis and forward-looking indicators",
        ],
        "content": (
            "## Credit Risk Appetite Framework\n\n"
            "### Purpose and Regulatory Context\n\n"
            "This framework establishes the firm's credit risk appetite for its commercial "
            "lending portfolio in accordance with PRA Supervisory Statement SS1/23 (Risk "
            "Appetite Frameworks) and the Basel Committee's Principles for Sound Credit Risk "
            "Assessment and Valuation. The risk appetite framework articulates the aggregate "
            "level and types of credit risk the firm is willing to assume within its risk "
            "capacity, in pursuit of its strategic objectives and business plan.\n\n"
            "### Risk Appetite Statement\n\n"
            "The firm seeks to maintain a diversified commercial lending portfolio that generates "
            "risk-adjusted returns above the cost of capital while preserving capital adequacy "
            "ratios with appropriate buffers above regulatory minima. The firm has no appetite "
            "for lending that: (a) cannot be adequately assessed using available data and "
            "methodologies; (b) creates unacceptable concentration to any single counterparty, "
            "sector, or geography; (c) does not meet minimum underwriting standards for "
            "borrower creditworthiness and collateral adequacy; or (d) exposes the firm to "
            "unquantifiable environmental, social, or governance risks.\n\n"
            "### Quantitative Risk Appetite Metrics\n\n"
            "The following metrics define the firm's quantitative credit risk appetite:\n\n"
            "- **Expected Loss (EL):** Portfolio EL not to exceed 1.2% of total exposure "
            "(amber at 1.0%, red at 1.2%)\n"
            "- **Non-Performing Loan Ratio:** NPL ratio not to exceed 4.0% of gross loans "
            "(amber at 3.0%, red at 4.0%)\n"
            "- **Provision Coverage Ratio:** Minimum 80% coverage of NPLs by specific provisions "
            "(amber at 90%, red at 80%)\n"
            "- **Risk-Weighted Asset Density:** RWA density not to exceed 55% of gross exposure "
            "(amber at 50%, red at 55%)\n"
            "- **Single Name Concentration:** No single counterparty exposure to exceed 10% of "
            "own funds without board approval (amber at 7%, red at 10%)\n"
            "- **Sector Concentration:** No single sector to exceed 25% of total portfolio "
            "(amber at 20%, red at 25%)\n\n"
            "### Stress Testing Integration\n\n"
            "Risk appetite metrics are calibrated against stress test outcomes from the firm's "
            "Internal Capital Adequacy Assessment Process (ICAAP). The firm must maintain capital "
            "ratios above PRA-prescribed buffers under a severe but plausible stress scenario "
            "equivalent to the Bank of England's Annual Cyclical Scenario. Risk appetite "
            "limits are set such that, under the stress scenario, the firm does not breach "
            "its Total Capital Requirement plus Pillar 2A and combined buffer requirements.\n\n"
            "### Governance and Escalation\n\n"
            "The Chief Risk Officer is accountable for maintaining and monitoring the risk "
            "appetite framework. Monthly risk appetite utilisation reports are presented to "
            "the Executive Risk Committee. Amber threshold breaches require documented "
            "remediation plans within 10 business days. Red threshold breaches trigger "
            "immediate escalation to the Board Risk Committee and may result in temporary "
            "lending restrictions pending corrective action."
        ),
    },
    {
        "policy_area": "concentration_limits",
        "title": "Portfolio Concentration Limits and Large Exposure Management Policy",
        "regulation_reference": "CRR Article 395, PRA Large Exposures",
        "effective_date": "2024-01-01",
        "key_requirements": [
            "No single counterparty exposure to exceed 25% of eligible capital (CRR Article 395 hard limit)",
            "Internal single-name limit of 10% of own funds requiring Board Risk Committee pre-approval above 7%",
            "Sector concentration limits of 25% of total portfolio exposure per NACE Level 1 sector",
            "Geographic concentration limits of 30% of total portfolio per UK region",
            "Product type concentration limits ensuring diversification across term loans, revolving credit, and trade finance",
            "Connected counterparty identification and aggregation in accordance with CRR Article 4(1)(39)",
        ],
        "compliance_checks": [
            "Daily large exposure report generated with automated breach detection and alerting",
            "Connected counterparty mapping reviewed quarterly against Companies House and beneficial ownership data",
            "Sector concentration dashboard updated weekly with forward-looking pipeline analysis",
            "Geographic heat map produced monthly showing regional exposure distribution",
            "Pre-deal concentration impact assessment completed for all new facilities above GBP 5 million",
            "Annual concentration risk stress test conducted as part of ICAAP",
        ],
        "content": (
            "## Portfolio Concentration Limits and Large Exposure Management\n\n"
            "### Regulatory Framework\n\n"
            "This policy implements the firm's compliance with the Capital Requirements "
            "Regulation (CRR) Article 395 large exposure limits and the PRA's supervisory "
            "expectations for concentration risk management in commercial lending portfolios. "
            "Concentration risk -- the risk of loss arising from insufficiently diversified "
            "portfolios -- is a material risk driver in commercial banking and has been a "
            "contributing factor in multiple bank failures. The PRA expects firms to manage "
            "concentration risk beyond the minimum CRR requirements through internal limits "
            "and active portfolio management.\n\n"
            "### Large Exposure Limits\n\n"
            "The CRR Article 395 imposes a hard limit of 25% of eligible capital for exposure "
            "to any single counterparty or group of connected clients. The firm operates with "
            "an internal limit of 10% of own funds, with exposures above 7% requiring pre-approval "
            "from the Board Risk Committee. Connected counterparty identification follows CRR "
            "Article 4(1)(39) criteria, supplemented by the firm's own assessment of economic "
            "dependency and control relationships. Beneficial ownership data from Companies "
            "House, the PSC Register, and commercial data providers is used to identify "
            "connections not apparent from corporate structure alone.\n\n"
            "### Sector Concentration Management\n\n"
            "The firm maintains sector concentration limits of 25% of total portfolio exposure "
            "per NACE Level 1 sector classification. Current portfolio allocation reflects a "
            "deliberate strategy of sector diversification, with the largest sector allocation "
            "(real estate and construction) capped at 22% of total exposure. Sector limits are "
            "reviewed annually against the firm's macroeconomic outlook and sector risk "
            "assessments. During periods of sectoral stress, tactical limits may be tightened "
            "below standing limits on CRO authority.\n\n"
            "The following sector-specific sub-limits apply:\n\n"
            "- **Construction:** Maximum 15% of total exposure (reflecting elevated default rates)\n"
            "- **Retail:** Maximum 12% of total exposure (reflecting structural sector challenges)\n"
            "- **Commercial Real Estate:** Maximum 20% of total exposure (PRA supervisory focus)\n"
            "- **Agriculture:** Maximum 8% of total exposure (subsidy transition risk)\n\n"
            "### Geographic Concentration\n\n"
            "Geographic concentration limits of 30% of total portfolio per UK region ensure "
            "diversification across economic geographies. The firm monitors exposure by NUTS 1 "
            "region and maintains enhanced monitoring for regions with concentrated industry "
            "profiles. London and the South East, while representing the largest single "
            "geographic allocation at approximately 28%, are considered lower risk due to "
            "economic diversification.\n\n"
            "### Monitoring and Reporting\n\n"
            "The Credit Risk function produces daily large exposure reports with automated "
            "breach detection integrated into the firm's risk management system. Concentration "
            "dashboards -- covering single-name, sector, geographic, and product dimensions -- "
            "are presented to the Executive Risk Committee monthly and the Board Risk Committee "
            "quarterly. Pre-deal concentration impact assessments are mandatory for all new "
            "facilities above GBP 5 million, ensuring that new business does not create or "
            "exacerbate limit breaches."
        ),
    },
]


def generate_regulatory_docs() -> list[dict]:
    """Generate 4 regulatory policy documents for RAG ingestion.

    Returns a list of structured dictionaries covering FCA Consumer Duty,
    fair lending guidelines, risk appetite framework, and concentration
    limits. Each contains metadata fields for filtering and markdown
    content for semantic search in Weaviate.
    """
    docs: list[dict] = []

    for policy in _REGULATORY_POLICIES:
        doc = {
            "document_id": str(uuid4()),
            "document_type": "regulatory_policy",
            "policy_area": policy["policy_area"],
            "title": policy["title"],
            "regulation_reference": policy["regulation_reference"],
            "effective_date": policy["effective_date"],
            "key_requirements": policy["key_requirements"],
            "compliance_checks": policy["compliance_checks"],
            "content": policy["content"],
            "generated_at": datetime.utcnow().isoformat(),
        }
        docs.append(doc)

    return docs
