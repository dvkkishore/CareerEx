# Security remediation notes

## Exposed credentials

Earlier versions of `app.py` committed database connection credentials to this public repository. Treat those credentials as compromised even though the current source uses environment variables. **Revoke or rotate them with the database provider before deploying this code.** Removing them from the working tree does not remove them from Git history, forks, or clones. Review historic commits and your provider's access logs, and consider a coordinated history rewrite only after confirming collaborators can migrate their clones.

The current source reads `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB` and `FLASK_SECRET_KEY` from environment variables. `.env.example` lists placeholder names only. The application reads the process environment directly; it does not automatically load a `.env` file. Set values in the host's secret/environment configuration or export them in the shell before starting the app. A missing required variable intentionally prevents startup.

## Unresolved security issues

The existing authentication path still compares plaintext passwords stored in the database. An earlier process-global `logged_in` flag was replaced by Flask's per-request session state. Treat the app as **not safe for public deployment** until password storage, account migration or resets, CSRF protection, and authentication/authorization tests have been addressed.

The application now references the tracked `FinalJob.csv` dataset, but the live MySQL installation and ML behavior remain unverified by the offline tests. No external database was accessed as part of the documented remediation.

If you believe credentials or personal data were accessed, review provider logs and consider appropriate incident-response obligations.
