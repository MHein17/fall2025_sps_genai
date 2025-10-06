print("Starting test...")

try:
    from helper_lib.trainer import train_vae_model
    print("Success! train_vae_model imported")
    print(f"Function: {train_vae_model}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete")