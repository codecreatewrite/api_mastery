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
    # URL: https://api.github.com/repos/{owner}/ rep}
    url = f"repos/{owner}/{repo}"
    repo_data = self.github_client.get(url)
    return repo_data

  def compare_repos(self, repo_list):
    # Compare multiple repositories
    # Show stars, forks, language, last update
    results = {}

    for rep in repo_list:
      owner, repo = rep.strip('/').split('/')
      data = self.get_repo_info(owner, repo)
      results[rep] = data

    for rep, data in results.items():
      print(f"{rep.title():15} | {data['data']['allow_forking']:10} | {data['data']['forks']:10} | {data['data']['language']:10} | {data['data']['stargazers_count']:10} | {data['data']['updated_at']:10}")

# Test with popular repos
analyzer = GitHubAnalyzer()
repos = [
    "microsoft/vscode",
    "facebook/react",
    "tensorflow/tensorflow"
]
analyzer.compare_repos(repos)
