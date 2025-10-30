# Security Setup Guide

This guide walks through setting up comprehensive security scanning for the MiniMacLLM repository.

## Overview

We use a defense-in-depth approach with multiple security scanning tools:

```text
┌─────────────────────────────────────────────────┐
│                 CODE SCANNING                    │
├─────────────────┬─────────────────┬─────────────┤
│     CodeQL      │     Bandit      │   Semgrep   │
│  (Semantic)     │  (Python AST)   │   (SAST)    │
└─────────────────┴─────────────────┴─────────────┘
                         │
┌─────────────────────────┴─────────────────────────┐
│              DEPENDENCY SCANNING                   │
├─────────────────┬─────────────────┬───────────────┤
│   Dependabot    │     Safety      │  pip-audit    │
│  (Updates)      │    (CVEs)       │  (Advisories) │
└─────────────────┴─────────────────┴───────────────┘
                         │
┌─────────────────────────┴─────────────────────────┐
│               SECRET SCANNING                      │
├─────────────────────────┬─────────────────────────┤
│      Gitleaks          │   GitHub Secret Scanner  │
│   (Custom rules)       │    (Built-in)           │
└────────────────────────┴─────────────────────────┘
```

## Step 1: Enable GitHub Security Features

1. Go to your repository settings
2. Navigate to **"Security & analysis"**
3. Enable the following:

   - ✅ **Dependency graph** - Required for Dependabot
   - ✅ **Dependabot alerts** - Get notified of vulnerable dependencies
   - ✅ **Dependabot security updates** - Auto-PRs for security fixes
   - ✅ **Code scanning alerts** - For CodeQL results
   - ✅ **Secret scanning alerts** - Detect exposed secrets

## Step 2: Configure Branch Protection

1. Go to **Settings → Branches**
2. Add rule for `main` branch
3. Configure protection rules:

```yaml
Required status checks:
  - CodeQL / Analyze Python Code
  - Security Scan / Bandit Security Scan
  - Security Scan / Safety Dependency Check
  - Security Scan / Semgrep SAST Scan
  - Security Scan / Secret Scanning

Required reviews:
  - Required approving reviews: 1
  - Dismiss stale reviews: Yes
  - Require review from CODEOWNERS: Yes (optional)

Other settings:
  - Require branches up to date: Yes
  - Include administrators: Yes (recommended)
  - Restrict push access: Yes (optional)
```

## Step 3: Configure Workflows

The following workflows are included:

### CodeQL Analysis (`.github/workflows/codeql.yml`)

- **Purpose**: Deep semantic code analysis
- **Runs**: On PRs, pushes to main, weekly schedule
- **Languages**: Python
- **Query suites**: security-extended, security-and-quality

### Security Scanning (`.github/workflows/security-scan.yml`)

Includes multiple tools:

1. **Bandit** - Python AST-based security linter
   - Detects: Hardcoded passwords, SQL injection, unsafe functions
   - Output: JSON and text reports

2. **Safety** - Dependency vulnerability scanner
   - Checks: requirements.txt against known CVEs
   - Database: Safety DB (updated regularly)

3. **Semgrep** - Pattern-based SAST
   - Rules: Auto-configured for Python
   - Custom rules: Can be added for ML-specific patterns

4. **Gitleaks** - Secret scanning
   - Prevents: Committing API keys, tokens
   - Config: Custom rules in `.gitleaks.toml`

### Dependabot (`.github/dependabot.yml`)

- **Updates**: Weekly for pip, monthly for GitHub Actions
- **Grouped**: Development and ML dependencies grouped
- **Auto-merge**: Can be configured for patch updates

## Step 4: Local Development Setup

### Install Security Tools Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install security tools
pip install bandit safety semgrep pip-audit

# Install pre-commit for Git hooks
pip install pre-commit
```

### Configure Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', '.bandit', '-r', 'src/', 'scripts/']

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: detect-private-key
```

Install hooks:

```bash
pre-commit install
pre-commit run --all-files  # Test run
```

## Step 5: Security Best Practices

### 1. Environment Variables

Never hardcode sensitive data. Use environment variables:

```python
# Bad
api_key = "sk-1234567890abcdef"

# Good
import os
api_key = os.environ.get("API_KEY")
```

### 2. Model Security

Verify model integrity:

```python
import hashlib

def verify_model_checksum(model_path, expected_hash):
    """Verify model file integrity before loading."""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    actual_hash = sha256_hash.hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"Model checksum mismatch!")

    return True

# Use weights_only=True when possible
model = torch.load(model_path, map_location='cpu', weights_only=True)
```

### 3. Input Validation

Always validate and sanitize inputs:

```python
def sanitize_text_input(text, max_length=10000):
    """Sanitize text input for model processing."""
    if not isinstance(text, str):
        raise ValueError("Input must be string")

    # Remove null bytes and control characters
    text = text.replace('\0', '')
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

    # Enforce length limit
    if len(text) > max_length:
        text = text[:max_length]

    return text
```

### 4. Resource Limits

Prevent resource exhaustion:

```python
import resource
import signal

def set_resource_limits():
    """Set resource limits for model inference."""
    # Limit CPU time to 60 seconds
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))

    # Limit memory to 8GB
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, -1))

    # Set timeout for operations
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minutes

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")
```

## Step 6: Monitoring and Response

### Security Alerts

1. **GitHub Security Tab**: Check regularly for alerts
2. **Email Notifications**: Enable for security alerts
3. **Slack/Discord Integration**: Use GitHub webhooks for real-time alerts

### Incident Response

1. **Assess**: Determine severity and scope
2. **Contain**: Disable affected features if needed
3. **Fix**: Develop and test patches
4. **Release**: Deploy fixes promptly
5. **Post-mortem**: Document and learn

## Step 7: Regular Maintenance

### Weekly Tasks

- Review Dependabot PRs
- Check security alerts
- Update dependencies: `pip list --outdated`

### Monthly Tasks

- Run full security audit:
  ```bash
  bandit -r src/ scripts/ -f html -o security-report.html
  safety check --full-report
  pip-audit
  ```

- Review and update security policies
- Check for new security best practices

### Quarterly Tasks

- Security training/awareness
- Review access controls
- Update threat model
- Penetration testing (if applicable)

## Common Issues and Solutions

### 1. False Positives

**Bandit**: Use inline comments to suppress false positives:
```python
# nosec B101 - This is a test assertion, not a security issue
assert user_input == expected_value
```

**Gitleaks**: Update `.gitleaks.toml` to exclude false positives

### 2. Vulnerable Dependencies

**Immediate**: If critical vulnerability:
```bash
# Pin to secure version immediately
pip install package==secure_version
pip freeze > requirements.txt
```

**Long-term**: Find alternatives or contribute fixes

### 3. Performance Impact

Security scanning can slow down CI/CD:
- Run comprehensive scans on schedule, not every commit
- Use incremental scanning where possible
- Parallelize security jobs

## Resources

- [GitHub Security Docs](https://docs.github.com/en/code-security)
- [OWASP Python Security](https://owasp.org/www-project-python-security/)
- [Bandit Docs](https://bandit.readthedocs.io/)
- [Semgrep Rules](https://semgrep.dev/explore)
- [CWE Top 25](https://cwe.mitre.org/top25/)

## Conclusion

Security is an ongoing process, not a one-time setup. This configuration provides:

- **Prevention**: Pre-commit hooks, code review
- **Detection**: Multiple scanning tools, alerts
- **Response**: Clear procedures, documentation
- **Improvement**: Regular updates, learning

Remember: The goal is not zero vulnerabilities, but managing risk effectively while maintaining development velocity.

---

*Last updated: October 2025*