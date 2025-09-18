import requests
import urllib.parse
import secrets
import hashlib
import base64
from typing import Dict, Optional
import webbrowser
import http.server
import socketserver
from threading import Thread

class OAuth2Client:
    """
    OAuth 2.0 client implementation (Authorization Code Flow)
    """
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str = "http://localhost:8080/callback"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        
        # OAuth endpoints (example for GitHub)
        self.authorization_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"
        
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
    
    def generate_pkce_challenge(self) -> tuple:
        """Generate PKCE code verifier and challenge for security"""
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        return code_verifier, code_challenge
    
    def get_authorization_url(self, scopes: list = None, state: str = None) -> str:
        """Generate authorization URL for user consent"""
        if state is None:
            state = secrets.token_urlsafe(32)
        
        if scopes is None:
            scopes = ['repo', 'user:email']  # GitHub example scopes
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(scopes),
            'state': state,
            'response_type': 'code'
        }
        
        return f"{self.authorization_url}?{urllib.parse.urlencode(params)}"
    
    def exchange_code_for_token(self, authorization_code: str) -> Dict:
        """Exchange authorization code for access token"""
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': authorization_code,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            response = requests.post(
                self.token_url,
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                
                return {
                    'success': True,
                    'access_token': self.access_token,
                    'token_type': token_data.get('token_type', 'bearer'),
                    'scope': token_data.get('scope', ''),
                    'refresh_token': self.refresh_token
                }
            else:
                return {
                    'success': False,
                    'error': f'Token exchange failed: {response.status_code}',
                    'details': response.text
                }
        
        except requests.RequestException as e:
            return {
                'success': False,
                'error': f'Request failed: {str(e)}'
            }
    
    def make_authenticated_request(self, url: str, method: str = 'GET', **kwargs) -> Dict:
        """Make authenticated API request using OAuth token"""
        if not self.access_token:
            return {
                'success': False,
                'error': 'No access token available. Please authenticate first.'
            }
        
        headers = kwargs.get('headers', {})
        headers.update({
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        })
        kwargs['headers'] = headers
        
        try:
            response = requests.request(method, url, **kwargs)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': 'Token expired or invalid'
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

class SimpleCallbackServer:
    """
    Simple HTTP server to handle OAuth callback
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.authorization_code = None
        self.server = None
        self.server_thread = None
    
    def start_server(self):
        """Start callback server"""
        handler = self._create_handler()
        
        with socketserver.TCPServer(("", self.port), handler) as httpd:
            self.server = httpd
            print(f"🌐 Callback server running on http://localhost:{self.port}")
            httpd.handle_request()  # Handle one request then stop
    
    def _create_handler(self):
        """Create request handler class"""
        server_instance = self
        
        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                # Parse the callback URL
                path_parts = self.path.split('?')
                if len(path_parts) > 1:
                    query_params = urllib.parse.parse_qs(path_parts[1])
                    
                    if 'code' in query_params:
                        server_instance.authorization_code = query_params['code'][0]
                        
                        # Send success response
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b'''
                        <html>
                        <body>
                            <h2>Authorization Successful!</h2>
                            <p>You can close this window and return to the application.</p>
                        </body>
                        </html>
                        ''')
                    
                    elif 'error' in query_params:
                        error = query_params['error'][0]
                        
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(f'''
                        <html>
                        <body>
                            <h2>Authorization Failed</h2>
                            <p>Error: {error}</p>
                        </body>
                        </html>
                        '''.encode())
                
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress log messages
                pass
        
        return CallbackHandler

# Complete OAuth 2.0 Flow Example
def github_oauth_flow():
    """
    Complete GitHub OAuth 2.0 authentication flow
    """
    
    # GitHub OAuth App credentials (you need to create these)
    CLIENT_ID = "your_github_oauth_app_client_id"
    CLIENT_SECRET = "your_github_oauth_app_client_secret"
    
    print("🔐 GitHub OAuth 2.0 Authentication")
    print("=" * 40)
    print("Note: You need to create a GitHub OAuth App first")
    print("Go to: GitHub Settings > Developer settings > OAuth Apps")
    print()
    
    if not CLIENT_ID.startswith('your_'):
        oauth_client = OAuth2Client(CLIENT_ID, CLIENT_SECRET)
        
        # Step 1: Get authorization URL
        auth_url = oauth_client.get_authorization_url(['repo', 'user:email'])
        print(f"🌐 Opening authorization URL in browser...")
        print(f"If it doesn't open automatically, visit: {auth_url}")
        
        # Open browser
        webbrowser.open(auth_url)
        
        # Step 2: Start callback server
        callback_server = SimpleCallbackServer()
        callback_server.start_server()
        
        # Step 3: Exchange code for token
        if callback_server.authorization_code:
            print("✅ Authorization code received")
            token_result = oauth_client.exchange_code_for_token(callback_server.authorization_code)
            
            if token_result['success']:
                print("🎉 OAuth authentication successful!")
                print(f"Access token: {token_result['access_token'][:20]}...")
                
                # Step 4: Make authenticated request
                user_result = oauth_client.make_authenticated_request(
                    'https://api.github.com/user'
                )
                
                if user_result['success']:
                    user_data = user_result['data']
                    print(f"👤 Authenticated as: {user_data['login']} ({user_data['name']})")
                    print(f"📧 Email: {user_data.get('email', 'Private')}")
                    return oauth_client
                else:
                    print(f"❌ Failed to get user info: {user_result['error']}")
            else:
                print(f"❌ Token exchange failed: {token_result['error']}")
        else:
            print("❌ No authorization code received")
    else:
        print("❌ Please configure your GitHub OAuth App credentials")
    
    return None

# Example usage
if __name__ == "__main__":
    oauth_client = github_oauth_flow()
    
    if oauth_client:
        print("\n🎯 You can now make authenticated API requests!")
        
        # Example: List user's repositories
        repos_result = oauth_client.make_authenticated_request(
            'https://api.github.com/user/repos',
            params={'per_page': 5}
        )
        
        if repos_result['success']:
            print("\n📂 Your repositories:")
            for repo in repos_result['data']:
                print(f"   - {repo['name']} ({repo['stargazers_count']} stars)")
