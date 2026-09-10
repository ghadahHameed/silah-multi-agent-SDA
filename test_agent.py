from product_agent import analyze_product


def run_test(
    test_number,
    product_name,
    product_description
):

    print(
        f"\n========== TEST {test_number} =========="
    )

    print(
        f"Product: {product_name}"
    )


    try:

        result = analyze_product(
            product_name,
            product_description
        )


        print(
            result.model_dump_json(
                indent=2
            )
        )


        print(
            f"TEST {test_number}: PASS"
        )


    except Exception as error:

        print(
            f"TEST {test_number}: FAIL"
        )

        print(
            error
        )


# CHANGE LATER:
# Replace these test products
# with realistic products for your project.

run_test(
    1,
    "Product A",
    "Add a realistic product description here."
)


run_test(
    2,
    "Product B",
    "Add a realistic product description here."
)


run_test(
    3,
    "Product C",
    "Add a realistic product description here."
)