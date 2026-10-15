# Security remediation notes

## Exposed credentials

Earlier versions of `app.py` committed database connection credentials to this public repository. Treat those credentials as compromised even after this branch is merged. **Revoke or rotate them with the database provider before deploying this code.** Removing them from the working tree does not remove them from Git history, forks, or clones. Review historic commits and your provider's access logs, and consider a coordinated history rewrite only after confirming collaborators can migrate their clones.

This branch moves `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB` and `FLASK_SECRET_KEY` into environment variables. `.env.example` lists placeholder names only. The application reads the process environment directly; it does not automatically load a `.env` file. Set values in the host's secret/environment configuration or export them in the shell before starting the app. A missing required variable intentionally prevents startup.

## Unresolved security issues

The existing authentication path compares plaintext passwords stored in the database, and the process-global `logged_in` flag is not a safe per-user authentication mechanism. Treat the existing app as **not safe for public deployment** until password storage, account migration or resets, session-based access control, CSRF protection, and tests have been addressed.

The app additionally references an untracked `jobs.csv` file, so these configuration changes alone do not establish a working installation. No external database was accessed while preparing this change.

If you believe credentials or personal data were accessed, review provider logs and consider appropriate incident-response obligations.
