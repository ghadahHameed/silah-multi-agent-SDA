from product_agent import analyze_product


def run_test(
    test_number,
    product_name,
    product_description
):

    print(f"\n========== TEST {test_number} ==========")
    print(f"Product: {product_name}")

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

        assert result.product_summary.strip() != ""
        assert len(result.ideal_customer_profile.target_industries) > 0
        assert len(result.ideal_customer_profile.preferred_company_size) > 0

        print(f"TEST {test_number}: PASS")

    except Exception as error:

        print(f"TEST {test_number}: FAIL")
        print(error)


run_test(
    1,
    "Cybersecurity Awareness Platform",
    "A B2B platform that helps organizations train employees on cybersecurity awareness, phishing risks, and security best practices."
)


run_test(
    2,
    "Inventory Management System",
    "A cloud-based B2B system that helps retailers and distributors track inventory, stock levels, suppliers, and warehouse operations."
)


run_test(
    3,
    "HR Recruitment Platform",
    "A B2B recruitment platform that helps companies manage job applications, screen candidates, and organize hiring workflows."
)
