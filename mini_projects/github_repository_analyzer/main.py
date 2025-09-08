#!/usr/bin/env python3
"""
Lesson: github_respositoty_analyzer
Created: 2025-09-03T17:57:47+01:00
Project: api_mastery
Template: script
"""
import requests
from api_mastery.clients.robust_api_client import RobustAPIClient

# Your challenge template
class GitHubAnalyzer:
  def __init__(self):
    # Initialize with robust API client
    self.github_client = RobustAPIClient("https://api.github.com")
    self.cache = {}

  def get_repo_info(self, owner, repo):
    # Get repository data from GitHub API
    # URL: https://api.github.com/repos/{owner}/{repo}
    url = f"repos/{owner}/{repo}"
    data = self.github_client.get(url)
    return data

  def compare_repos(self, repo_list):
    # Compare multiple repositories
    # Show stars, forks, language, last update
    data_list = []
    for rep_string in repo_list:
      if '/' in rep_string:
        owner, repo = rep_string.strip('/').split('/', 1)
        results = self.get_repo_info(owner, repo)

        if results['success']:
          raw = results['data']

          repo_data = {
            'name': raw[].get('name'),
            'stars': raw.get('stargazers_count'),
            'forks': raw.get('forks_count'),
            'language': raw.get('language'),
            'create_date': raw.get('created_at'),
            'last_update': raw.get('updated_at')
          }
          data_list.append(repo_data)

        elif results['error'] == 'Rate limited':
          print(f"Retry after: {results['retry_after']}")

        else:
          print(f"Unable to get data for {rep_string}")

      else:
        print(f"Skipping {rep_string}...")
        continue

    self.display_repos(data_list)

  def display_repos(self, data_list):
    if data_list:
      print(f"{'Fullname':>25} | {'Stars':>15} | {'Forks':>15} | {'Language':>15} | {'Last update':10}")
      for r_d in data_list:
        print(f"{str(r_d.get('fullname') or 'N/A'):25} | {str(r_d.get('stars') or 'N/A'):15} | {str(r_d.get('forks') or 'N/A'):15} | {str(r_d.get('language') or 'N/A'):15} | {str(r_d.get('last_update') or 'N/A')}")
        print()
    else:
      print("No data to display")

# Test with popular repos
analyzer = GitHubAnalyzer()
repos = [
    "microsoft/vscode",
    "facebook/react",
    "tensorflow/tensorflow"
]
analyzer.compare_repos(repos)
