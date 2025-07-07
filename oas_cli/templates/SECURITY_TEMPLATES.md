# Security Agent Templates

This directory contains security agent templates for the Open Agent Stack framework. These templates demonstrate advanced security use cases, multi-engine support, and comprehensive behavioral contracts.

## Available Security Templates

### 1. Security Threat Analyzer (`security-threat-analyzer.yaml`)

**Purpose:** Analyzes security events and identifies potential threats with detailed classification.

**Key Features:**
- **Engine:** Claude/Anthropic (configurable)
- **Threat Detection:** Comprehensive threat type classification
- **Evidence-Based:** High confidence scoring and indicator analysis
- **Validation:** Strict behavioral contracts with false positive prevention

**Use Cases:**
- SOC (Security Operations Center) event analysis
- Automated threat hunting
- Security alert triage
- Incident detection pipeline

**Input:** Security events (failed logins, port scans, data access attempts)
**Output:** Threat classification, severity, indicators, confidence scores

### 2. Security Risk Assessor (`security-risk-assessor.yaml`)

**Purpose:** Evaluates security threats and determines organizational risk with comprehensive mitigation recommendations.

**Key Features:**
- **Engine:** Claude/Anthropic (configurable)
- **Business Focus:** Financial impact assessment and business context
- **Actionable Output:** Immediate, short-term, and long-term recommendations
- **Compliance:** Regulatory and compliance impact analysis

**Use Cases:**
- Enterprise risk management
- Security investment prioritization
- Executive reporting and dashboards
- Compliance and audit support

**Input:** Threat analysis results + business context
**Output:** Risk scores, business impact, mitigation strategies, escalation guidance

### 3. Security Incident Responder (`security-incident-responder.yaml`)

**Purpose:** Comprehensive incident response coordination with multi-step analysis and automated response workflows.

**Key Features:**
- **Engine:** OpenAI GPT-4 (demonstrates multi-engine flexibility)
- **Multi-Task:** Incident analysis, response coordination, and reporting
- **Workflow Integration:** Designed for DACP orchestration
- **Stakeholder Management:** Executive and technical communication

**Use Cases:**
- Automated incident response
- Crisis management coordination
- Post-incident analysis and reporting
- Compliance documentation

**Input:** Incident descriptions, status updates, investigation findings
**Output:** Response plans, coordination guidance, executive reports

## Multi-Engine Support

These templates demonstrate Open Agent Stack's multi-engine capabilities:

- **Claude/Anthropic:** `security-threat-analyzer.yaml`, `security-risk-assessor.yaml`
- **OpenAI:** `security-incident-responder.yaml`
- **Local/Custom:** Templates easily configurable for any engine

### Engine Configuration Examples

**Claude/Anthropic:**
```yaml
intelligence:
  type: llm
  engine: anthropic
  model: claude-3-5-sonnet-20241022
  config:
    temperature: 0.3
    max_tokens: 1000
```

**OpenAI:**
```yaml
intelligence:
  type: llm
  engine: openai
  model: gpt-4
  endpoint: https://api.openai.com/v1
  config:
    temperature: 0.3
    max_tokens: 2000
```

**Local:**
```yaml
intelligence:
  type: llm
  engine: local
  model: llama2-7b
  config:
    temperature: 0.3
    max_tokens: 1000
```

## Behavioral Contracts

All security templates include advanced behavioral contracts demonstrating:

### Security-Specific Flags
```yaml
behavioural_flags:
  conservatism: "high"           # Conservative security posture
  evidence_based: "strict"       # Require evidence for conclusions
  business_focus: "critical"     # Business-aligned recommendations
  urgency_awareness: "critical"  # Time-sensitive response handling
```

### Validation Rules
```yaml
validation_rules:
  confidence_threshold: 0.3      # Minimum confidence for assertions
  min_indicators: 1              # Require threat indicators
  timeline_realism: true         # Realistic timeline estimates
```

### Safety Checks
```yaml
safety_checks:
  false_positive_prevention: true # Prevent false threat alerts
  recommendation_safety: true     # Safe mitigation recommendations
  compliance_aware: true          # Compliance-conscious outputs
```

## Agent-to-Agent Workflows

These templates are designed for multi-agent security workflows:

### Sequential Analysis Pipeline
1. **Threat Analyzer** → Identifies threats from security events
2. **Risk Assessor** → Evaluates business risk and impact
3. **Incident Responder** → Coordinates response activities

### DACP Workflow Configuration
```yaml
workflows:
  security_analysis_pipeline:
    steps:
      - agent: security-threat-analyzer
        task: analyze_security_event
        route_output_to:
          agent: security-risk-assessor
          task: assess_security_risk
          input_mapping:
            threat_type: "{{output.threat_type}}"
            threat_severity: "{{output.threat_severity}}"
            # ... additional mappings
```

## Getting Started

### 1. Generate Security Agents
```bash
# Generate threat analyzer (Claude-powered)
oas init --spec security-threat-analyzer.yaml --output threat-analyzer/

# Generate risk assessor (Claude-powered)  
oas init --spec security-risk-assessor.yaml --output risk-assessor/

# Generate incident responder (OpenAI-powered)
oas init --spec security-incident-responder.yaml --output incident-responder/
```

### 2. Configure API Keys
```bash
# For Claude agents
echo "ANTHROPIC_API_KEY=your-key-here" > threat-analyzer/.env
echo "ANTHROPIC_API_KEY=your-key-here" > risk-assessor/.env

# For OpenAI agents
echo "OPENAI_API_KEY=your-key-here" > incident-responder/.env
```

### 3. Run Security Analysis
```python
# Example: Threat analysis
threat_result = threat_analyzer.analyze_security_event(
    event_type="login_failure",
    event_details="15 failed login attempts for admin account",
    source_ip="192.168.1.100",
    timestamp="2024-01-15T14:30:00Z"
)

# Example: Risk assessment
risk_result = risk_assessor.assess_security_risk(
    threat_identified=threat_result['threat_identified'],
    threat_type=threat_result['threat_type'],
    threat_severity=threat_result['threat_severity'],
    threat_indicators=threat_result['threat_indicators'],
    analysis_summary=threat_result['analysis_summary'],
    asset_context="Critical customer database with PII data"
)
```

## Production Considerations

### Security Requirements
- **API Key Management:** Use secure key storage (AWS Secrets Manager, Azure Key Vault)
- **Network Security:** Deploy in secure network segments with proper access controls
- **Audit Logging:** Enable comprehensive logging for compliance and forensics
- **Data Privacy:** Ensure sensitive data handling compliance (GDPR, HIPAA, etc.)

### Scalability
- **Load Balancing:** Deploy multiple agent instances for high availability
- **Queue Management:** Use message queues for high-volume event processing
- **Caching:** Implement result caching for frequently analyzed patterns
- **Monitoring:** Set up comprehensive monitoring and alerting

### Integration Patterns
- **SIEM Integration:** Connect with Splunk, QRadar, Sentinel for event ingestion
- **Ticketing Systems:** Integrate with ServiceNow, Jira for incident management
- **Communication:** Connect with Slack, Teams for stakeholder notifications
- **Orchestration:** Use DACP for complex multi-agent workflows

## Advanced Features

### Custom Threat Intelligence
Extend templates with organization-specific threat intelligence:
```yaml
threat_intelligence:
  internal_indicators: true
  external_feeds: ["vendor1", "vendor2"]
  custom_rules: "./custom_rules.yaml"
```

### Regulatory Compliance
Configure templates for specific compliance requirements:
```yaml
compliance_frameworks:
  - "SOX"
  - "GDPR" 
  - "HIPAA"
  - "PCI-DSS"
```

### Integration APIs
Templates support standard security APIs:
- **STIX/TAXII** for threat intelligence sharing
- **MITRE ATT&CK** for threat classification
- **OpenIOC** for indicator formats
- **SCAP** for vulnerability assessment

## Contributing

When contributing new security templates:

1. **Follow Security Best Practices:** Implement appropriate behavioral contracts
2. **Include Documentation:** Provide clear use case descriptions and examples
3. **Test Thoroughly:** Validate with realistic security scenarios
4. **Consider Compliance:** Address regulatory and audit requirements
5. **Multi-Engine Support:** Test with different LLM engines

## Learn More

- **Open Agent Stack:** [openagentstack.ai](https://openagentstack.ai)
- **DACP Orchestration:** [GitHub - DACP](https://github.com/aswhitehouse/dacp)
- **Behavioral Contracts:** Framework documentation and validation patterns
- **Security Use Cases:** Real-world implementations and case studies