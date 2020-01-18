"""This module generates a dataset for the LightTag platform
"""
import asyncio

from ucla_topic_analysis.data.coroutines.light_tag import LightTagDataSetPipeline

if __name__ == "__main__":
    asyncio.run(LightTagDataSetPipeline.generate_dataset())