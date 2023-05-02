def main():
    if len(sys.argv) < 3:
        print("Usage: script.py base_model other_model [other_models...]")
        sys.exit(1)

    file1 = Path(sys.argv[1])
    files = sys.argv[2:]

    seed = 114514
    torch.manual_seed(seed)
    print(f"seed: {seed}")

    print(f"Loading base model: {file1}")
    model_a = ModelEvaluator.load_model(file1)
    print("Base model loaded")
    calc_a = CrossAttnCalculator(model_a)

    print()
    print(f"base: {file1.name} [{ModelEvaluator.model_hash(file1)}]")
    print()

    map_attn_a = {}
    map_rand_input = {}
    for n in range(3, 11):
        hidden_dim, embed_dim = model_a[
            f"model.diffusion_model.output_blocks.{n}.1.transformer_blocks.0.attn1.to_q.weight"
        ].shape
        rand_input = torch.randn([embed_dim, hidden_dim])

        map_attn_a[n] = calc_a.eval(n, rand_input)
        map_rand_input[n] = rand_input

    del model_a

    for file2 in files:
        file2 = Path(file2)
        model_b = ModelEvaluator.load_model(file2)
        calc_b = CrossAttnCalculator(model_b)
        sims = []
        for n in range(3, 11):
            attn_a = map_attn_a[n]
            attn_b = calc_b.eval(n, map_rand_input[n])

            sim = torch.mean(torch.cosine_similarity(attn_a, attn_b))
            sims.append(sim)

        print(
            f"{file2} [{ModelEvaluator.model_hash(file2)}] - {torch.mean(torch.stack(sims)) * 1e2:.2f}%"
        )
        print()

if __name__ == "__main__":
    main()
