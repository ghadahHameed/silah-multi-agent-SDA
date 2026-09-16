from product_agent import analyze_product


def run_test(
    test_number,
    product_name,
    product_description,
    expected_status,
    preferred_city=None,
    requested_company_count=None
):

    print(
        f"\n========== TEST {test_number} =========="
    )

    print(
        f"Product: {product_name}"
    )

    try:

        result = analyze_product(
            product_name=product_name,
            product_description=product_description,
            sender_name="Aseel Alsaad",
            sender_company_name="Silah Demo Company",
            sender_email="aseel@example.com",
            sender_phone="+966500000000",
            preferred_city=preferred_city,
            requested_company_count=requested_company_count
        )

        assert result.status == expected_status, \
            "Unexpected agent status"

        assert result.sender_info.name, \
            "Sender name is missing"

        assert result.search_preferences.requested_company_count >= 1, \
            "Invalid requested company count"

        if result.status == "ready":

            assert result.icp_industries, \
                "ICP industries are empty"

            assert result.buyer_profile.strip(), \
                "Buyer profile is empty"

        if result.status == "needs_clarification":

            assert not result.icp_industries, \
                "Industries should be empty"

            assert result.clarification_question, \
                "Clarification question is missing"

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

        print(error)


run_test(
    1,
    "Cybersecurity Awareness Platform",
    """
    A B2B platform that trains employees
    on cybersecurity awareness, phishing risks,
    and security best practices.
    """,
    expected_status="ready",
    preferred_city="Riyadh",
    requested_company_count=5
)


run_test(
    2,
    "Inventory Management System",
    """
    A B2B platform that helps retailers
    and distributors manage stock,
    suppliers, and warehouse operations.
    """,
    expected_status="ready",
    preferred_city="Jeddah",
    requested_company_count=10
)


run_test(
    3,
    "Smart Solution",
    """
    A solution that helps businesses work better.
    """,
    expected_status="needs_clarification"
)
