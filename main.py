from src.config import config
from src.cli import get_parser, load_params
from src.handlers import command_handlers

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
        # try:
        #     handler(args)
        # except Exception as e:
        #     print(f'Exception in {handler.__name__}:')
        #     parser.error(str(e))
        #     raise e
    else:
        parser.error(f"Unknown command: {args.command}")



