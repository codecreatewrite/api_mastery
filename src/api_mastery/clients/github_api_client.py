class GitHubAPIClient:
    """
    Professional GitHub API client with full authentication and rate limiting
    """

    def __init__(self, token: Optional[str] = None):
        self.base_url = "https://api.github.com"
        self.token = token
        self.session = requests.Session()

        # Set up authentication if token provided
        if token:
            self.session.headers.update({
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Python-GitHub-Client/1.0'
            })

        # Rate limiting (authenticated: 5000/hour, unauthenticated: 60/hour)
        self.rate_limiter = RateLimiter(
            calls_per_second=1.0 if not token else 1.4  # Conservative rates
        )

        # Track API usage
        self.api_calls_made = 0
        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated GitHub API request with rate limiting"""

        self.rate_limiter.wait_if_needed()

        try:
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=30
            )

            # Update rate limit tracking
            self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
            self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
            self.api_calls_made += 1

            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json(),
                    'rate_limit_remaining': self.rate_limit_remaining
                }

            elif response.status_code == 403 and 'rate limit' in response.text.lower():
                reset_time = datetime.fromtimestamp(self.rate_limit_reset)
                return {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'reset_time': reset_time,
                    'wait_seconds': max(0, self.rate_limit_reset - time.time())
                }

            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': 'Invalid or missing GitHub token'
                }

            elif response.status_code == 404:
                return {
                    'success': False,
                    'error': 'Repository not found or private'
                }

            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.text[:200]}'
                }

        except requests.RequestException as e:
            return {
                'success': False,
                'error': f'Request failed: {str(e)}'
            }

    def get_repository(self, owner: str, repo: str) -> Dict:
        """Get repository information"""
        return self._make_request(f"repos/{owner}/{repo}")

    def get_repository_languages(self, owner: str, repo: str) -> Dict:
        """Get repository programming languages"""
        return self._make_request(f"repos/{owner}/{repo}/languages")

    def get_repository_contributors(self, owner: str, repo: str, limit: int = 10) -> Dict:
        """Get repository contributors"""
        return self._make_request(f"repos/{owner}/{repo}/contributors", {'per_page': limit})

    def get_repository_commits(self, owner: str, repo: str, since: str = None, limit: int = 100) -> Dict:
        """Get repository commits"""
        params = {'per_page': min(limit, 100)}
        if since:
            params['since'] = since

        return self._make_request(f"repos/{owner}/{repo}/commits", params)

    def get_repository_issues(self, owner: str, repo: str, state: str = 'open', limit: int = 100) -> Dict:
        """Get repository issues"""
        params = {
            'state': state,
            'per_page': min(limit, 100)
        }
        return self._make_request(f"repos/{owner}/{repo}/issues", params)

    def get_user_repositories(self, username: str, repo_type: str = 'public', limit: int = 30) -> Dict:
        """Get user's repositories"""
        params = {
            'type': repo_type,
            'sort': 'updated',
            'per_page': min(limit, 100)
        }
        return self._make_request(f"users/{username}/repos", params)

    def search_repositories(self, query: str, sort: str = 'stars', limit: int = 30) -> Dict:
        """Search repositories"""
        params = {
            'q': query,
            'sort': sort,
            'order': 'desc',
            'per_page': min(limit, 100)
        }
        return self._make_request("search/repositories", params)

    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status"""
        result = self._make_request("rate_limit")
        if result['success']:
            return result['data']
        return {'error': result['error']}
