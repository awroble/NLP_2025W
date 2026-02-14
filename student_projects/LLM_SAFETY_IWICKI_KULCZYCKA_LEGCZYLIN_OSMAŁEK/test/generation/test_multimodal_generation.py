from src.generation.MultimodalTemplateFiller import MultimodalTemplateFiller


def test():
    MultimodalTemplateFiller.prepare_templates("sensitive-data-extraction")
    MultimodalTemplateFiller.prepare_templates("mental-physical-health")
    MultimodalTemplateFiller.prepare_templates("social-engineering")

    MultimodalTemplateFiller.generate_prompts(
        filename_stem="sensitive-data-extraction",
        n=40,
        neutral_frac=0.2  # 20% safe (neutral), 80% unsafe
    )
    MultimodalTemplateFiller.generate_prompts(
        filename_stem="mental-physical-health",
        n=40,
        neutral_frac=0.2  # 20% safe (neutral), 80% unsafe
    )
    MultimodalTemplateFiller.generate_prompts(
        filename_stem="social-engineering",
        n=40,
        neutral_frac=0.2  # 20% safe (neutral), 80% unsafe
    )


if __name__ == '__main__':
    test()
