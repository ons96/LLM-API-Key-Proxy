**AGENTS.md**
================

**Introduction**
---------------

This document outlines the role, mission, and operating procedures for the autonomous coding agent responsible for updating the project's README and deploying it to Replit.

**Role/Mission**
----------------

As an autonomous coding agent, your mission is to:

1. Update the README with setup instructions, environment variables, and directions for adding new source plugins.
2. Create and deploy a Procfile or start script to enable Replit deployment using Uvicorn.
3. Make decisions independently to achieve the above objectives while utilizing only free resources.
4. Document any questions or uncertainties in the QUESTIONS.md file for further investigation.

**Technical Stack**
-------------------

* Programming language: Python
* Web framework: FastAPI (via Uvicorn)
* Deployment platform: Replit
* CI/CD pipeline: GitHub Actions

**Requirements**
---------------

1. Python 3.9+ installed on the system
2. Uvicorn installed via pip (`pip install uvicorn[standard]`)
3. GitHub Actions workflow file updated to utilize only free resources
4. QUESTIONS.md file available for storing any questions or uncertainties

**File Structure**
------------------

The project file structure should be maintained as follows:
```markdown
main.py
app.py
README.md
requirements.txt
Procfile
QUESTIONS.md
.github/workflows/main.yml
```
**Testing Requirements**
------------------------

No specific testing requirements are specified for this task. However, it is essential to ensure that the README and Procfile are correctly formatted and functional.

**Git Protocol**
----------------

* Clone the repository using `git clone <repository-url>`
* Navigate to the project directory using `cd <project-name>`
* Make changes to the README and Procfile files
* Commit the changes using `git add .` and `git commit -m "<commit-message>"`
* Push the changes to the remote repository using `git push origin <branch-name>`

**Completion Criteria**
------------------------

The completion criteria for this task are as follows:

1. The README is updated with setup instructions, environment variables, and directions for adding new source plugins.
2. A Procfile or start script is created to enable Replit deployment using Uvicorn.
3. The GitHub Actions workflow file is updated to utilize only free resources.
4. Any questions or uncertainties are documented in the QUESTIONS.md file.

**Decision-Making and Self-Management**
-----------------------------------------

As an autonomous coding agent, you should make decisions independently to achieve the project's objectives. If faced with any questions or uncertainties, please document them in the QUESTIONS.md file and seek assistance when necessary. Remember to utilize only free resources to ensure minimal costs and maximum efficiency.