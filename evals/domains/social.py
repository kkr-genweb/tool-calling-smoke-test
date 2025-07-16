"""
Social Media Domain (Twitter)
=============================

Mock social media API for testing agent interactions with social platforms.
"""

from typing import Any, Dict, List

from pydantic_ai import RunContext

from ..base import BaseDomainClient, CallLogger


class TwitterAPIClient(BaseDomainClient):
    """Mock Twitter API client."""
    
    def __init__(self, logger: CallLogger) -> None:
        super().__init__(logger)
        self.posts: List[str] = []

    @property
    def domain_name(self) -> str:
        return "twitter"

    async def post_tweet(self, text: str) -> Dict[str, Any]:
        """Post a tweet."""
        self.posts.append(text)
        return self.logger.log("twitter", "post_tweet", {"text": text}, {"tweet_id": len(self.posts), "ok": True})

    async def retweet(self, tweet_id: int) -> bool:
        """Retweet a tweet by ID."""
        return self.logger.log("twitter", "retweet", {"tweet_id": tweet_id}, True)

    async def comment(self, tweet_id: int, text: str) -> bool:
        """Comment on a tweet."""
        return self.logger.log("twitter", "comment", {"tweet_id": tweet_id, "text": text}, True)


def register_social_tools(agent, deps_type):
    """Register social media tools with the agent."""
    
    @agent.tool
    async def tweet_post(ctx: RunContext[deps_type], text: str) -> Dict[str, Any]:
        """Post a tweet."""
        return await ctx.deps.twitter.post_tweet(text)

    @agent.tool
    async def tweet_retweet(ctx: RunContext[deps_type], tweet_id: int) -> bool:
        """Retweet a tweet."""
        return await ctx.deps.twitter.retweet(tweet_id)

    @agent.tool
    async def tweet_comment(ctx: RunContext[deps_type], tweet_id: int, text: str) -> bool:
        """Comment on a tweet."""
        return await ctx.deps.twitter.comment(tweet_id, text)