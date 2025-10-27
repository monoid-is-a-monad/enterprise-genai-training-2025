# Contributing Guidelines

**Enterprise GenAI Training Program**  
**Provided by:** ADC ENGINEERING & CONSULTING LTD

## Purpose

This document outlines the standards and procedures for contributing to the Enterprise GenAI Training Program materials.

## Content Standards

### Code Quality
- **Python Code:**
  - Follow PEP 8 style guidelines
  - Include docstrings for all functions and classes
  - Add type hints for function parameters and return values
  - Ensure code is tested and runnable
  - Include requirements.txt or environment.yml

- **Notebooks:**
  - Clear cell organization with markdown headers
  - Explanatory text before code cells
  - Output cells showing expected results
  - References to documentation and resources

### Documentation Standards
- **Markdown Files:**
  - Use consistent heading hierarchy (H1 for titles, H2 for sections)
  - Include table of contents for documents > 200 lines
  - Add links to related resources
  - Include examples and code snippets where appropriate

- **Slide Decks:**
  - Follow the template in `templates/slides/`
  - Maximum 30 slides per hour of content
  - Include speaker notes
  - Add references and citations

### Lab Exercises
- **Structure:**
  - Clear learning objectives
  - Prerequisites section
  - Step-by-step instructions
  - Verification/testing section
  - Solution provided separately
  - Estimated completion time

## File Naming Conventions

### General Rules
- Use lowercase with hyphens: `file-name.md`
- Include date for reports: `YYYY-MM-DD-report-name.md`
- Version control: `document-v1.0.md`

### Specific Patterns
- **Slides:** `week-XX-topic-slides.pdf`
- **Labs:** `week-XX-lab-topic.ipynb`
- **Solutions:** `week-XX-lab-topic-solution.ipynb`
- **Reports:** `YYYY-MM-month-N-report.md`

## Version Control

### Branch Strategy
- `main` - Production-ready materials
- `develop` - Integration branch for new content
- `feature/week-XX-topic` - Feature branches for specific weeks

### Commit Messages
Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New training material
- `fix`: Corrections or bug fixes
- `docs`: Documentation updates
- `style`: Formatting changes
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(week-01): add GenAI fundamentals slides
fix(week-03): correct OpenAI API authentication example
docs(readme): update progress tracking section
```

## Review Process

### Material Review Checklist
- [ ] Technical accuracy verified
- [ ] Code examples tested and working
- [ ] Links and references checked
- [ ] Spelling and grammar reviewed
- [ ] Follows style guidelines
- [ ] Learning objectives clearly stated
- [ ] Prerequisites documented
- [ ] Estimated time included

### Peer Review
- All materials should be reviewed by at least one other trainer
- Use pull requests for reviews
- Address all comments before merging

## Quality Assurance

### Testing
- **Code Examples:** Run all code snippets to ensure they work
- **Labs:** Complete labs from start to finish
- **Links:** Verify all external links are accessible
- **Dependencies:** Test with clean environment setup

### Accessibility
- Use descriptive alt text for images
- Ensure sufficient color contrast in slides
- Provide transcripts for video content
- Use clear, simple language

## Feedback & Improvements

### Collecting Feedback
- Weekly retrospectives with trainees
- Anonymous feedback forms
- Issue tracking for problems
- Monthly review sessions

### Continuous Improvement
- Update materials based on feedback
- Track common questions and add to FAQs
- Maintain changelog for major updates
- Archive outdated content

## Contact

For questions about these guidelines, contact the training coordinator.

---

**Version:** 1.0  
**Last Updated:** October 27, 2025
