#!/usr/bin/env python3
"""
Lesson: github_respositoty_analyzer
Created: 2025-09-03T17:57:47+01:00
Project: api_mastery
Template: script
"""
import requests
from playground.robustapiclient import RobustAPIClient

# Your challenge template
class GitHubAnalyzer:
  def __init__(self):
    # Initialize with robust API client
    self.github_client = RobustAPIClient("https://api.github.com/")
    self.cache = {}

  def get_repo_info(self, owner, repo):
    # Get repository data from GitHub API
    # URL: https://api.github.com/repos/{owner}/{repo}
    url = f"repos/{owner}/{repo}"
    repo_data = self.github_client.get(url)
    return repo_data

  def compare_repos(self, repo_list):
    # Compare multiple repositories
    # Show stars, forks, language, last update
    results = {}

    for repo in repo_list:
      data = get_repo_info(owner, repo)
      results[repo] = data

    for repo, data in results.items():
      print(f"{repo.title():15}")

# Test with popular repos
analyzer = GitHubAnalyzer()
repos = [
    "microsoft/vscode",
    "facebook/react",
    "tensorflow/tensorflow"
]
analyzer.compare_repos(repos)