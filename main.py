# main.py
import sys
from pipeline import RedditPipeline
from config import REQUEST_SIZE


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <interval_minutes>")
        print("Example: python main.py 5")
        sys.exit(1)

    interval = int(sys.argv[1])
    pipeline = RedditPipeline()

    pipeline.run_pipeline(REQUEST_SIZE)
    pipeline.start_automation(interval)

    try:
        pipeline.interactive_mode()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()