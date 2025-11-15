import os

def process_file(input_path, output_path):
    """
    Reads a file line by line, keeps only lines starting with '1',
    replaces the leading '1' with '80', and writes output.
    """
    
    processed_lines = []

    with open(input_path, "r") as f:
        for line in f:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Only keep lines that start with '1'
            if stripped.startswith("4 "):               # YOLO-style
                new_line = "80 " + stripped[2:]        # replace "1 "
                processed_lines.append(new_line)

            elif stripped.startswith("4"):              # general case
                new_line = "80" + stripped[1:]
                processed_lines.append(new_line)

            # Any other line is skipped (deleted)

    # Save output file
    with open(output_path, "w") as f:
        for line in processed_lines:
            f.write(line + "\n")

    print(f"Processed: {os.path.basename(input_path)} → {os.path.basename(output_path)}")


def process_folder(input_folder, output_folder):
    """
    Processes all .txt files in a folder.
    """
    # Create output folder if missing
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)

            process_file(in_path, out_path)

    print("All files processed.")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    input_folder = "data/test/labels_x"   # change to your folder
    output_folder = "{}_".format(input_folder)  # will be created

    process_folder(input_folder, output_folder)
