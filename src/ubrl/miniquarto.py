def qmd_code_blocks(qmd_src: str) -> list[str]:
    blocks = []
    in_block = False
    current_block = []
    block_header = ''
    for lino, line in enumerate(qmd_src.split('\n')):
        if line.strip().startswith('```{python}'):
            in_block = True
            block_header = '\n' * lino 
        elif line.strip().startswith('```'):
            if in_block:
                in_block = False
                blocks.append(block_header + '\n'.join(current_block))
                current_block = []
            else:
                print(f"Skipping block beginning with {line!r}")
        elif in_block:
            current_block.append(line)
    return blocks


def run_qmd(filename: str):
    with open(filename, 'r') as f:
        qmd_src = f.read()
    blocks = qmd_code_blocks(qmd_src)
    context = dict()
    for block in blocks:
        print('Running:')
        print(block)
        code = compile(block, filename, 'exec')
        exec(code, context, context)


if __name__ == '__main__':
    import sys
    run_qmd(sys.argv[1])
