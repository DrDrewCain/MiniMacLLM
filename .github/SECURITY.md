# Security Policy

## Supported Versions

Currently supported versions for security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of MiniMacLLM seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do NOT** Create a Public Issue

Security vulnerabilities should not be reported through public GitHub issues.

### 2. Report Privately

Please report security vulnerabilities by emailing:
- **Email**: hello@voxmatch.org
- **Subject**: [SECURITY] MiniMacLLM Vulnerability Report

### 3. Include in Your Report

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Your contact information

### 4. Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution Timeline**: Depends on severity
  - Critical: 1-2 weeks
  - High: 2-4 weeks
  - Medium/Low: 4-8 weeks

## Security Measures

### Code Scanning

This repository uses multiple security scanning tools:

1. **GitHub CodeQL**: Advanced semantic code analysis
   - Runs on all PRs and pushes to main
   - Weekly scheduled scans
   - Security-extended query suite enabled

2. **Bandit**: Python-specific security linter
   - Scans for common security issues in Python code
   - Checks for hardcoded passwords, SQL injection risks, etc.

3. **Safety**: Dependency vulnerability scanner
   - Checks all Python dependencies for known CVEs
   - Runs on every PR

4. **Semgrep**: SAST (Static Application Security Testing)
   - Pattern-based scanning for security anti-patterns
   - Custom rules for ML/LLM specific vulnerabilities

5. **Gitleaks**: Secret scanning
   - Prevents committing API keys, tokens, passwords
   - Scans entire git history

6. **Dependabot**: Automated dependency updates
   - Weekly scans for dependency vulnerabilities
   - Automatic PRs for security patches

### Best Practices

1. **No Secrets in Code**
   - Use environment variables for sensitive data
   - Never commit API keys, passwords, or tokens
   - Use `.env` files (git-ignored) for local development

2. **Dependency Management**
   - Pin dependency versions in requirements.txt
   - Regularly update dependencies
   - Review Dependabot PRs promptly

3. **Code Review**
   - All PRs require review before merging
   - Security-sensitive changes need thorough review
   - Use branch protection rules

4. **Model Security**
   - Models are stored via Git LFS
   - Verify model checksums before loading
   - Never load untrusted model files

## Known Security Considerations

### 1. Model Loading

⚠️ **Warning**: Loading untrusted PyTorch models can execute arbitrary code.

**Mitigation**:
- Only load models from trusted sources
- Verify checksums before loading
- Consider using `torch.load(..., weights_only=True)` when possible

### 2. Tokenizer Security

The BPE tokenizer can potentially be exploited with adversarial inputs.

**Mitigation**:
- Input validation and sanitization
- Length limits on input text
- Rate limiting for API usage

### 3. Resource Exhaustion

Large models can consume significant memory/compute.

**Mitigation**:
- Set resource limits
- Implement timeouts
- Monitor resource usage

## Security Configuration

### Recommended GitHub Settings

1. **Enable Security Features**:
   - Go to Settings → Security & analysis
   - Enable all security features:
     - ✅ Dependency graph
     - ✅ Dependabot alerts
     - ✅ Dependabot security updates
     - ✅ Code scanning alerts
     - ✅ Secret scanning alerts

2. **Branch Protection Rules** (for `main` branch):
   - ✅ Require PR reviews before merging
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require status checks to pass:
     - CodeQL
     - Security Scan / Bandit
     - Security Scan / Safety
     - Security Scan / Semgrep
     - Tests (if configured)
   - ✅ Require branches to be up to date
   - ✅ Include administrators
   - ✅ Restrict who can push to main

### Local Development Security

```bash
# Install pre-commit hooks for local scanning
pip install pre-commit
pre-commit install

# Run security scans locally
bandit -r src/ scripts/
safety check
semgrep --config=auto src/
```

## Disclosure Policy

When we receive a security report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible
5. Credit the reporter (unless they prefer to remain anonymous)

## Credits

We appreciate the security research community and will acknowledge reporters who help us keep MiniMacLLM secure.

## Contact

- **Security Email**: hello@voxmatch.org
- **GitHub Security Advisories**: [Enable in repo settings]
- **Bug Bounty**: Currently not offered, but we appreciate responsible disclosure

---

*This security policy is adapted from best practices recommended by GitHub and the Open Source Security Foundation.*