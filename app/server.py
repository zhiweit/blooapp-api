import sys
import uvicorn

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from fastapi import FastAPI, APIRouter
from app.models.image_request import ImageRequest
from app.models.image_response import ImageResponse
from app.dependencies import model

load_dotenv()

app = FastAPI()
router = APIRouter(prefix="/api")


prompt = """
You are an expert on answering questions briefly and accurately about recycling in Singapore.
What is/are the item(s) in the image and can they be recycled?
If you are not sure what the item is in the image, say you are not sure whether the item is what you think it is, but provide a list of maximum 5 possible items that you think this item is.
If they can be recycled, please provide clear and concise recycling instructions on how I can recycle them properly.
Return the response in JSON format.
Example of a valid response with single item identified that is recyclable:
{
    "item": {
        "name": "Paper Milk Carton",
        "description": "The item in the image is a paper milk carton, specifically a 1-liter container for low-fat fresh milk.",
        "recyclable": 'true',
        "instructions": "In Singapore, paper milk cartons like this can be recycled, but it's important to prepare them correctly because they are often lined with plastic or aluminum. Here’s how to recycle this item properly:

        1. **Empty the Carton**: Ensure the carton is completely empty of any milk.
        2. **Rinse the Carton**: Rinse it out to remove any milk residue, as residue can contaminate other recyclables.
        3. **Dry the Carton**: Allow the carton to dry to prevent mold growth in the recycling bin.
        4. **Flatten the Carton**: Flatten the carton to save space in your recycling bin and facilitate easier transportation and processing.
        5. **Recycling Bin**: Place the clean, dry, and flattened carton in the recycling bin designated for paper or comingled recyclables, depending on your local recycling guidelines.

        By following these steps, you help ensure that the carton is recycled efficiently and does not contaminate other recyclable materials."
    },
    "other_items": []
}

Example of a valid response with single item identified that is not recyclable:
{
    "item": {
        "name": "Sheet Mask",
        "description": "The item in the image is a sheet mask, typically used for skincare. These masks are generally made from a lightweight fabric-like material that is infused with various skincare serums.",
        "recyclable": 'false',
        "instructions": "In Singapore, sheet masks are not recyclable through regular municipal recycling programs due to their composition and contamination with cosmetic products. Here’s what you can do:

        1. **Dispose of Properly**: Since the sheet mask itself and the serum it contains can contaminate other recyclables, it should be disposed of in the general waste bin.
        2. **Check the Packaging**: If the sheet mask came in a separate packaging, such as a plastic wrapper or paper box, check those for recycling symbols. Clean and dry them before recycling if they meet the local recycling guidelines.
        3. **Reduce Waste**: To minimize waste, consider using reusable face masks or those made from biodegradable materials if available.
        4. **Special Recycling Programs**: Occasionally, cosmetic brands or stores offer recycling programs specifically for beauty product packaging. Inquire at the place of purchase or directly with the brand to see if they provide such a service.

        Always refer to the latest guidelines from the National Environment Agency (NEA) of Singapore for the most up-to-date information on waste management practices."
    },
    "other_items": []
}

Example of a valid response with single item identified that is partially recyclable:
{
    "item": {
        "name": "Cosmetic container",
        "description": "The item in the image is a cosmetic container, specifically for a facial cream with a pump dispenser.",
        "recyclable": 'partial',
        "instructions": "In Singapore, cosmetic containers made of plastic can be recycled, but you should separate the components because they often include different materials. Here's how to recycle this item properly:

        1. **Empty the Container**: Make sure the container is completely empty of any product.
        2. **Separate Components**: Detach the pump dispenser from the bottle, as the pump often contains metal springs and other non-recyclable components.
        3. **Clean the Container**: Rinse the plastic container to remove any residual product.
        4. **Dry the Container**: Allow it to air dry to avoid moisture in the recycling bin.
        5. **Discard Non-recyclable Parts**: Dispose of the pump in general waste, unless your local recycling program specifies that it can be recycled.
        6. **Recycle the Plastic Part**: Place the clean and dry plastic container in the recycling bin designated for plastics.

        It's important to follow these steps to ensure that the recyclable parts are processed correctly and to reduce contamination in the recycling stream."
    },
    "other_items": []
}

Example of a valid response with multiple items in the image:
{
    "item": {
        "name": "Shirt",
        "description": "The item in the image is a white long-sleeve shirt hanging on a plastic hanger.",
        "recyclable": 'false',
        "instructions": "If the shirt is in good condition, it is best to donate it to a charity or second-hand store. If it is worn out, some recycling programs accept textiles where they can be turned into industrial rags, insulation, or other textile byproducts. Check with local textile recycling facilities or drop-off points.
    },
    "other_items": [
        {
            "name": "Plastic Hanger",
            "description": "The item in the image is a plastic hanger.",
            "recyclable": 'false',
            "instructions": "Plastic hangers are typically not recyclable through curbside recycling programs due to their size, shape, and the mixed plastics they are often made from. Consider donating usable hangers to thrift stores or returning them to dry cleaners. If they are broken, they should be disposed of in the general waste unless a specific recycling option is available."
        }
    ]
    
}

"""


@router.get("/")
async def redirect_root_to_docs():
    return {"message": "Hello World"}
    # return RedirectResponse("/docs")


@router.post("/vision")
async def chat(request: ImageRequest):
    # get the base64 image from the request body
    base64_image = request.base64_image
    structured_model = model.with_structured_output(ImageResponse, method="json_mode")

    msg = await structured_model.ainvoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            ),
        ]
    )
    # print(msg)

    return msg


app.include_router(router)
if __name__ == "__main__":
    dev_mode = "dev" in sys.argv
    print(f"Running in {'development' if dev_mode else 'production'} mode")
    uvicorn.run("app.server:app", host="0.0.0.0", port=8080, reload=dev_mode)
