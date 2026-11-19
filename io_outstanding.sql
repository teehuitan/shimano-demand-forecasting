SELECT
    MaterialCode,
    Product_Type_Code,
    Model_Code,
    Plan_Model_Code,
    Quantity,
    Ship_To_Code,
    SLH_Order,
    Order_Status_Text,
    [Order_Entry_Date Date],
    [Customer_Request_Date Date],
    [Cancel_Date Date]
FROM Warehouse_AI.dbo.IncomingOrder_GoogleLooker
WHERE Plant = :plant
    AND _source_file = :source_file
