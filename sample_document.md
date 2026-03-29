# Acme Corp Employee Handbook 2026

## 1. Company Overview

### 1.1 Mission Statement

Acme Corp's mission is to deliver innovative cloud-based solutions that empower businesses to scale efficiently. Founded in 2015 in Austin, Texas, we have grown from a 5-person startup to a 2,000-employee global organization serving over 10,000 enterprise customers across 45 countries.

### 1.2 Core Values

Our five core values guide every decision we make:

1. **Customer Obsession** — We start with the customer and work backwards. Every product decision must trace back to a real customer need validated through research.
2. **Radical Transparency** — We share financials, roadmaps, and challenges openly with all employees through monthly town halls and the internal dashboard.
3. **Continuous Learning** — Every employee receives a $3,000 annual learning budget for courses, conferences, and certifications.
4. **Sustainable Innovation** — We commit to carbon-neutral operations by 2027 and factor environmental impact into all product decisions.
5. **Inclusive Excellence** — We believe diverse teams build better products. Our hiring panels must include at least two people from underrepresented groups.

### 1.3 Company History

- **2015**: Founded by Sarah Chen and Marcus Williams with $500K seed funding.
- **2017**: Launched CloudSync v1.0, our flagship product. Reached 100 customers.
- **2019**: Series B funding of $45M led by Sequoia Capital. Opened London office.
- **2021**: Reached 5,000 customers. Launched AI-powered analytics suite.
- **2023**: IPO on NASDAQ at $28 per share. Revenue crossed $500M.
- **2025**: Acquired DataFlow Inc. for $200M. Employee count reached 2,000.

## 2. Employment Policies

### 2.1 Work Arrangements

Acme Corp operates on a **hybrid work model**. Employees may work remotely up to 3 days per week. The required in-office days are Tuesday and Thursday, designated as collaboration days. Remote-first exceptions are available for roles explicitly tagged as "Remote OK" in the job posting.

Employees must maintain core hours of **10:00 AM to 3:00 PM** in their local time zone for synchronous collaboration. Outside core hours, employees have flexibility to structure their workday as they prefer.

International employees follow their local office policies, but must have at least a 4-hour overlap with their team's primary time zone.

### 2.2 Paid Time Off (PTO)

Full-time employees receive PTO based on tenure:

| Tenure | Annual PTO Days | Sick Days | Personal Days |
|--------|----------------|-----------|---------------|
| 0-2 years | 15 days | 10 days | 3 days |
| 3-5 years | 20 days | 10 days | 5 days |
| 6-10 years | 25 days | 12 days | 5 days |
| 10+ years | 30 days | 15 days | 7 days |

PTO must be requested through the HR portal at least 2 weeks in advance for absences of 3+ consecutive days. Unused PTO up to 5 days rolls over to the next year. PTO beyond the rollover limit expires on December 31st.

### 2.3 Parental Leave

Acme Corp provides **16 weeks of fully paid parental leave** for all new parents regardless of gender. An additional 4 weeks of unpaid leave may be taken. Parental leave can be taken within 12 months of the child's birth or adoption date. Employees may choose to take leave in two non-consecutive blocks with manager approval.

### 2.4 Performance Reviews

Performance reviews occur **twice per year** in June and December. The review process includes:

1. **Self-assessment** — Employee fills out accomplishments, challenges, and goals.
2. **Peer feedback** — Each employee selects 3-5 peers to provide 360-degree feedback.
3. **Manager assessment** — Direct manager evaluates against role-specific competencies.
4. **Calibration** — Leadership team calibrates ratings across departments to ensure fairness.
5. **Compensation review** — Ratings directly influence annual compensation adjustments in January.

Ratings use a 5-point scale: Exceptional (5), Exceeds Expectations (4), Meets Expectations (3), Needs Improvement (2), Unsatisfactory (1). Employees rated 2 or below are placed on a 90-day Performance Improvement Plan (PIP).

## 3. Compensation and Benefits

### 3.1 Salary Structure

Acme Corp uses a transparent salary framework with published salary bands for every role. Bands are benchmarked annually against the 75th percentile of industry data from Radford and Levels.fyi.

Each role has three salary tiers:
- **Entry**: 80-90% of band midpoint
- **Mid**: 90-110% of band midpoint
- **Senior**: 110-125% of band midpoint

Geographic adjustments are applied based on cost-of-living data. Tier 1 cities (San Francisco, New York, London) receive full band rates. Tier 2 cities receive 90%. Tier 3 and remote locations receive 80%.

### 3.2 Equity Compensation

All full-time employees receive equity grants as part of their total compensation package:

- **Initial Grant**: Awarded at hire, vesting over 4 years with a 1-year cliff.
- **Refresh Grants**: Annual refresh grants are awarded based on performance ratings. Employees rated "Exceeds Expectations" or above receive refresh grants equal to 25-50% of their initial grant value.
- **RSU Settlement**: RSUs settle quarterly on the 15th of March, June, September, and December.

### 3.3 Health Insurance

Acme Corp offers three health insurance plans through Blue Cross Blue Shield:

1. **Basic Plan**: $0 employee premium, $2,000 deductible, 80/20 coinsurance. Covers employee only.
2. **Standard Plan**: $150/month employee premium, $1,000 deductible, 90/10 coinsurance. Covers employee + dependents.
3. **Premium Plan**: $300/month employee premium, $500 deductible, 95/5 coinsurance. Covers employee + dependents. Includes vision and dental.

All plans include mental health coverage with up to 20 therapy sessions per year at no copay. Employees can change plans during the annual open enrollment period in November or within 30 days of a qualifying life event.

### 3.4 401(k) Retirement Plan

Acme Corp matches employee 401(k) contributions dollar-for-dollar up to **6% of base salary**. The company match vests over 3 years: 33% after year 1, 66% after year 2, and 100% after year 3. Employees may contribute up to the IRS annual limit. The plan offers 15 fund options including target-date funds, index funds, and a self-directed brokerage window.

## 4. Engineering Practices

### 4.1 Development Workflow

All engineering teams follow a **trunk-based development** model:

1. Create a short-lived feature branch from `main`.
2. Keep branches under 200 lines of changes when possible.
3. Open a pull request with at least 2 reviewers.
4. All CI checks must pass: unit tests, integration tests, linting, and security scanning.
5. Squash-merge into `main` after approval.
6. Deployments to production happen automatically via the CD pipeline every 2 hours.

Code review turnaround SLA is **4 business hours**. If a reviewer hasn't responded within the SLA, the author may add a backup reviewer and proceed.

### 4.2 On-Call and Incident Response

Engineering teams participate in an on-call rotation covering 24/7 support:

- **On-call shifts**: 1 week per rotation, with a primary and secondary on-call engineer.
- **Response SLA**: P1 incidents require acknowledgment within 15 minutes. P2 within 1 hour. P3 within 4 hours.
- **Incident commander**: For P1 incidents, a senior engineer assumes the IC role to coordinate response.
- **Post-mortem**: Every P1 and P2 incident requires a blameless post-mortem within 5 business days. The post-mortem must include: timeline, root cause, impact assessment, and action items with owners and deadlines.
- **Compensation**: On-call engineers receive a $500 weekly stipend plus $200 per incident page outside business hours.

### 4.3 Technology Stack

Acme Corp's approved technology stack includes:

- **Backend**: Python (FastAPI), Go (for high-performance services), Node.js (BFF layer)
- **Frontend**: React with TypeScript, Next.js for SSR applications
- **Databases**: PostgreSQL (primary), Redis (caching), ClickHouse (analytics)
- **Infrastructure**: AWS (primary cloud), Terraform (IaC), Kubernetes (orchestration)
- **Observability**: Datadog (metrics/traces), PagerDuty (alerting), Sentry (error tracking)
- **AI/ML**: Python, PyTorch, MLflow for model management, SageMaker for training

Teams requiring non-approved technologies must submit a Technology Adoption Proposal (TAP) to the Architecture Review Board.

### 4.4 Data Handling and Security

All customer data is classified into three tiers:

- **Tier 1 (Public)**: Marketing materials, public documentation. No special handling.
- **Tier 2 (Internal)**: Internal metrics, non-PII customer usage data. Encrypted at rest.
- **Tier 3 (Restricted)**: PII, financial data, authentication credentials. Encrypted at rest and in transit. Access requires role-based authorization and audit logging. Data must not leave production environments without DLP review.

All employees must complete annual security training. Engineers with access to Tier 3 data must additionally pass the Secure Coding Certification.

## 5. Travel and Expense Policy

### 5.1 Business Travel

Employees must book travel through the company's Navan portal. Booking policies:

- **Flights**: Economy class for trips under 6 hours. Premium economy for 6-10 hours. Business class for trips over 10 hours or with VP approval.
- **Hotels**: Up to $250/night in Tier 1 cities. Up to $175/night in other locations. Exceptions require manager pre-approval.
- **Meals**: Up to $75/day for individual meals. Team dinners up to $100/person with director approval.
- **Ground transportation**: Uber/Lyft for trips under $50. Rental car for trips where driving is more practical.

### 5.2 Expense Reimbursement

Expenses must be submitted within **30 days** of being incurred. Receipts are required for all expenses over $25. The approval chain is:

- Under $500: Direct manager approval
- $500-$2,000: Director approval
- $2,000-$10,000: VP approval
- Over $10,000: CFO approval

Reimbursements are processed within 10 business days and included in the next payroll cycle.

## 6. Learning and Development

### 6.1 Training Budget

Every employee receives a **$3,000 annual learning budget** that can be used for:

- Online courses (Coursera, Udemy, LinkedIn Learning, etc.)
- Industry conferences and workshops
- Professional certifications
- Technical books and subscriptions
- Language learning programs

Unused budget does not roll over. Requests over $1,000 require manager pre-approval. Employees must share a brief learning summary with their team after completing any training over $500.

### 6.2 Internal Mentorship Program

The mentorship program pairs junior employees with senior leaders for 6-month cycles. Mentors and mentees meet bi-weekly for 1-hour sessions. The program focuses on career development, technical skills, and leadership growth. Applications open in January and July each year.

### 6.3 Promotion Criteria

Promotions are considered during the June and December review cycles. Criteria include:

1. **Sustained performance**: At least two consecutive "Exceeds Expectations" ratings.
2. **Scope expansion**: Demonstrated ability to operate at the next level for 3+ months.
3. **Peer recognition**: Positive feedback from cross-functional collaborators.
4. **Business impact**: Measurable contribution to team or company OKRs.

Skip-level promotions are rare and require VP sponsorship and a calibration committee review.
